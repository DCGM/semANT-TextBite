# semANT-TextBite

Tool for extracting logical chunks from complex document pages.

Install from PyPi:
```
pip install textbite
```

Run by simply providing a folder of images alongside a folder of corresponding PAGE XMLs (such as obtained from pero-ocr):

```
textbite \
    --xml xmls/ \
    --img imgs/ \
    --yolo yolo.pt \
    --gnn gnn.pth \
    --normalizer norm.pkl \
    --save out/ \
    [--logging-level INFO] \
    [--json] \
    [--alto altos/]
```

By default, TextBite downloads neccessary models from the internet. In case you are working in an offline environment, you can [download it](https://nextcloud.fit.vutbr.cz/s/6jNgze6fLYXQBgq) yourself and provide path as an argument.
