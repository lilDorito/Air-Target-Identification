# scripts/

`check_no_data.py` blocks commits containing data-like files. To run it automatically on every commit:

```bash
printf '#!/bin/sh\npython scripts/check_no_data.py\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

Also: **clear notebook outputs that show raw rows** before committing. Embedded outputs should be aggregate tables and figures only.
