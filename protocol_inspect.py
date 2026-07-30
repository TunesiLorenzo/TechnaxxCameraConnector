from androguard.core.dex import DEX

dex = DEX(open("goplus_probe/classes.dex", "rb").read())

for cls in dex.get_classes():
    if "CamWrapper" in cls.get_name():
        print("\nCLASS", cls.get_name())
        for field in cls.get_fields():
            name = field.get_name()
            if any(x in name.lower() for x in ("batt", "port", "command")):
                encoded = field.get_init_value()
                try:
                    value = encoded.get_value() if encoded is not None else None
                except AttributeError:
                    value = encoded
                print(" FIELD", name, field.get_descriptor(), repr(value))

    if "MainViewController" not in cls.get_name():
        continue

    for method in cls.get_methods():
        code = method.get_code()
        if code is None:
            continue
        instructions = list(code.get_bc().get_instructions())
        outputs = [f"{item.get_name():24} {item.get_output()}" for item in instructions]
        matches = [
            index
            for index, output in enumerate(outputs)
            if "ShowBattery" in output
        ]
        for match in matches:
            print(
                "\nBATTERY METHOD",
                cls.get_name(),
                method.get_name(),
                method.get_descriptor(),
            )
            interesting = sorted(
                set(range(0, min(80, len(outputs))))
                | set(range(max(0, match - 75), min(len(outputs), match + 12)))
            )
            previous = -2
            for index in interesting:
                if index != previous + 1:
                    print(" ...")
                print(f" {index:04}: {outputs[index]}")
                previous = index
