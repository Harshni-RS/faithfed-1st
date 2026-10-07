# fix_width.py  (run from the project root: python fix_width.py)
import glob

for f in ["app.py"] + glob.glob("pages/*.py"):
    with open(f, encoding="utf-8") as fh:
        text = fh.read()
    new = (text.replace("use_container_width=True", 'width="stretch"')
               .replace("use_container_width=False", 'width="content"'))
    if new != text:
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(new)
        print("updated", f)