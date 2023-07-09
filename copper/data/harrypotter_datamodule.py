from typing import Optional
from pathlib import Path
import torch
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken
import requests
import numpy as np

class HarryPotterDataset(Dataset):
    def __init__(self, data_dir="data/", block_size=8, download=True, output_file="all_harry_potter_books.txt"):
        super().__init__()

        self.block_size = block_size
        self.data_dir = data_dir

        if download:
            Path(self.data_dir).mkdir(exist_ok=True, parents=True)

            filenames = ["Book 1 - The Philosopher's Stone.txt","Book 2 - The Chamber of Secrets.txt","Book 3 - The Prisoner of Azkaban.txt",
             "Book 4 - The Goblet of Fire.txt","Book 5 - The Order of the Phoenix.txt","Book 6 - The Half Blood Prince.txt",
             "Book 7 - The Deathly Hallows.txt"]

            def download_file(url, filename):
                response = requests.get(url, stream=True)
                with (Path(self.data_dir) / filename).open("wb") as fd:
                    for chunk in response.iter_content(chunk_size=1024):
                        fd.write(chunk)

            for filename in filenames:
                url = "https://github.com/formcept/whiteboard/raw/master/nbviewer/notebooks/data/harrypotter/" + filename
                download_file(url, filename)
            print("All Harry Potter Files Downloaded!!")

            def concatenate_files(filenames, output_file):
                with (Path(self.data_dir) / output_file).open("w") as outfile:
                    for filename in filenames:
                        with (Path(self.data_dir) / filename).open("r") as infile:
                            outfile.write(infile.read())

            concatenate_files(filenames, output_file)
            print("All Harry Potter Files Concatenated!! Master File Name: {}".format(output_file))

        with (Path(self.data_dir) / output_file).open() as f:
            self.text = f.read()

        cl100k_base = tiktoken.get_encoding("cl100k_base")

        # In production, load the arguments directly instead of accessing private attributes
        # See openai_public.py for examples of arguments for specific encodings
        self.encoder = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="cl100k_im",
            pat_str=cl100k_base._pat_str,
            mergeable_ranks=cl100k_base._mergeable_ranks,
            special_tokens={
                **cl100k_base._special_tokens,
                "<|im_start|>": 100264,
                "<|im_end|>": 100265,
            }
        )

        self.data = np.array(self.encoder.encode(self.text))

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = torch.from_numpy(
            (self.data[idx:idx + self.block_size]).astype(np.int64)
        )

        y = torch.from_numpy(
            (self.data[idx+1:idx+1+self.block_size]).astype(np.int64)
        )

        return x, y

class HarryPotterDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_ratio=0.7,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        block_size=8,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_train = None
        self.data_val = None
        self.block_size = block_size

    def prepare_data(self):
        HarryPotterDataset(self.hparams.data_dir, download=True, block_size=self.block_size)

    def setup(self, stage: Optional[str] = None):
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val:
            dataset = HarryPotterDataset(self.hparams.data_dir, download=False, block_size=self.block_size)

            self.data_train, self.data_val = random_split(
                dataset=dataset,
                lengths=[self.hparams.train_ratio, (1-self.hparams.train_ratio)],
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )