# semANT-TextBite

Tool for extracting logical chunks from complex document pages.

Install from PyPi:
```
pip install textbite
```

Run by simply providing a folder of images alongside a folder of corresponding PAGE XMLs (such as obtained from pero-ocr):

```
textbite --xml-input page_xmls/ --images jpegs/ --xml-output textbite-out/ [--model best-weights.pt]
```

By default, TextBite downloads a detection model from the internet. In case you are working in an offline environment, you can [download it](https://nextcloud.fit.vutbr.cz/s/5xHQcgosNai9pwa) yourself and provide path as an argument.
