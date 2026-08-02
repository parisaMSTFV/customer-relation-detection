# Data provenance

Every order, account, address, family token, relation group, and label is generated locally by `src/customer_relation_detection/synthetic.py` with NumPy seed 42.

The generator creates fictional buildings and units, assigns one or more accounts to each unit, emits repeated address observations, and introduces controlled formatting differences, abbreviations, Unicode digits, missing postcodes, and street typos. Buildings are assigned wholly to train, validation, or test so that threshold selection and final evaluation never share a location.

No names, addresses, customer identifiers, order history, query logic, schema, threshold, or result was copied from an employer or production system.
