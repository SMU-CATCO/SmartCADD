import os
import subprocess

from .data import Compound


class IterableDataset(object):
    def __init__(self, root_dir: str, batch_size: int = 1000):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.files = self._get_files(self.root_dir)
        self.len = self._count_total_lines() // self.batch_size

    def __len__(self):
        return self.len

    def __iter__(self):
        batch = []
        for file in self.files:
            with open(os.path.join(self.root_dir, file)) as f:
                for line in f:
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                    else:
                        line = line.strip().split()
                        batch.append(
                            Compound(smiles=line[0], id=line[1]),
                        )

    def _count_total_lines(self):
        result = subprocess.run(
            f"wc -l {self.root_dir}/*.smi | awk 'END{{print $1}}'",
            shell=True,
            stdout=subprocess.PIPE,
        )
        total_lines = int(result.stdout)
        return total_lines

    def _get_files(self, root_dir: str):
        files = sorted([f for f in os.listdir(root_dir) if f.endswith(".smi")])
        return files
