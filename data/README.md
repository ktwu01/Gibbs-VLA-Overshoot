# Data

This directory holds **symlinks** to local copies of benchmark datasets. Do not commit large files.

## Language-Table

After downloading the dataset to a path on your machine:

```bash
ln -s /path/to/your/language-table ./language-table
```

## CALVIN

```bash
ln -s /path/to/your/calvin/dataset ./calvin
```

Verify:

```bash
ls -la data/
```

Expected layout (example):

```text
data/
├── README.md
├── language-table -> /path/to/language-table
└── calvin -> /path/to/calvin
```
