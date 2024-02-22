### f2py calls

```bash
python -m numpy.f2py radau.f --overwrite-signature -m radau -h radau.pyf
# python -m numpy.f2py --overwrite-signature radau_decsol.f -m radau -h radau_decsol.pyf
```

```bash
python -m numpy.f2py -c radau.pyf radau.f decsol.f dc_decsol.f
# python -m numpy.f2py -c radau_decsol.pyf radau_decsol.f dc_decsol.f
```