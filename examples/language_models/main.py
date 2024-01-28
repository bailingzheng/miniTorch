import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from examples.language_models.data import CharDataset, InfiniteDataLoader
from models import Bigram, MLP, Transformer
from optim import AdamW


def create_datasets(input_file):

    # preprocessing of the input text file
    with open(input_file, "r") as f:
        words = f.read().splitlines()

    words = [w.strip() for w in words]
    words = [w for w in words if w]

    chars = sorted(list(set("".join(words))))
    max_word_length = max(len(w) for w in words)

    # partition the input data into a training and the test set
    M = min(1000, int(len(words) * 0.1)) # 10% of the training set, or up to 1000 examples
    rp = torch.randperm(len(words)).tolist()
    train_words = [words[i] for i in rp[:-M]]
    test_words = [words[i] for i in rp[-M:]]
    print(f"split up the dataset into {len(train_words)} training examples and {len(test_words)} test examples")

    # wrap in dataset objects
    train_dataset = CharDataset(train_words, chars, max_word_length)
    test_dataset = CharDataset(test_words, chars, max_word_length)

    return train_dataset, test_dataset


@torch.inference_mode()
def evaluate(model, dataset, batch_size=50, max_batches=None):
    model.eval()

    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    losses = []
    for i, batch in enumerate(loader):
        x, y = [t for t in batch]
        y_pred = model(x)

        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), ignore_index=-1)
        losses.append(loss.item())
        if max_batches is not None and i >= max_batches:
            break
    mean_loss = torch.tensor(losses).mean().item()

    model.train()
    return mean_loss


def generate(model, idx, temperature=1.0, do_sample=False, top_k=None):
    """Take a conditioning sequence of indices idx (LongTensor of shape (N, S)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.

    """
    for _ in range(model.block_size):

        logits = model.forward(idx)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def print_samples(num=10):
    """ samples from the model and pretty prints the decoded samples """

    X_init = torch.zeros(num, 1, dtype=torch.long)
    X_samp = generate(model, X_init, do_sample=True)

    print('-'*80)

    for i in range(X_samp.size(0)):
        row = X_samp[i, 1:].tolist() # note: we need to crop out the first <START> token

        # token 0 is the <STOP> token, so we crop the output sequence at that point
        crop_index = row.index(0) if 0 in row else len(row)
        row = row[:crop_index]
        word_samp = train_dataset.decode(row)
        print(word_samp)
    
    print('-'*80)


if __name__ == "__main__":

    # parse command line args
    parser = argparse.ArgumentParser(description="Language Model")

    # system
    parser.add_argument(
        "--input-file", 
        type=str, 
        default="examples/language_models/names.txt", 
        help="input file with things one per line"
    )
    parser.add_argument(
        "--num-workers", 
        type=int, 
        default=4, 
        help="number of data workers for both train/test"
    )

    # model
    parser.add_argument(
        "--type", 
        type=str, 
        default="transformer", 
        help="model class type to use, bigram | mlp | transformer"
    )
    parser.add_argument(
        "--num-layers", 
        type=int, 
        default=4, 
        help="number of hidden layers"
    )
    parser.add_argument(
        "--nhead", 
        type=int, 
        default=4, 
        help="number of heads in the multihead attention models"
    )
    parser.add_argument(
        "--num-features", 
        type=int, 
        default=64, 
        help="number of expected features in the input"
    )
    
    # optimization
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="batch size during optimization"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-3, 
        help="learning rate"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=1000, 
        help="max number of optimization steps to run for, or -1 for infinite."
    )
    
    args = parser.parse_args()
    torch.manual_seed(3407)

    train_dataset, test_dataset = create_datasets(args.input_file)
    vocab_size = train_dataset.get_vocab_size()
    block_size = train_dataset.get_block_size()
    print(f"{vocab_size=} | {block_size=}")

    if args.type == "bigram":
        model = Bigram(vocab_size, block_size)
    elif args.type == "mlp":
        model = MLP(vocab_size, block_size, args.num_features)
    elif args.type == "transformer":
        model = Transformer(vocab_size, block_size, args.num_features)
    else:
        raise ValueError(f"model type {args.type} is not recognized.")

    print(f"model #params: {sum(p.numel() for p in model.parameters())}")

    # init optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)

    # init dataloader
    batch_loader = InfiniteDataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        num_workers=args.num_workers
    )

    # training loop
    best_loss = None
    step = 0
    while True:

        t0 = time.time()

        x, y = batch_loader.next()
        y_pred = model(x)
        loss = F.cross_entropy(y_pred.view(-1, y_pred.size(-1)), y.view(-1), ignore_index=-1)

        model.zero_grad()
        loss.backward()
        optimizer.step()

        t1 = time.time()

        # logging
        if step % 100 == 0:
            print(f"step {step} | loss {loss.item():.4f} | step time (ms) {(t1-t0)*1000:.2f}")

        # evaluate the model
        if step > 0 and step % 500 == 0:
            train_loss = evaluate(
                model, 
                train_dataset, 
                batch_size=100, 
                max_batches=10
            )
            test_loss  = evaluate(
                model, 
                test_dataset,  
                batch_size=100, 
                max_batches=10
            )
            print(f"step {step} train loss: {train_loss:.4f} test loss: {test_loss:.4f}")

        step += 1
        # termination conditions
        if args.max_steps >= 0 and step >= args.max_steps:
            break
    
    print_samples(num=10)