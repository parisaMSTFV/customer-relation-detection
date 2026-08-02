# Data

`synthetic_orders.csv` contains generated order and address observations. `synthetic_ground_truth.csv` contains generated account-group labels, building-level splits, relation-signal types, and noise severity used only for evaluation.

Identifiers beginning with `ACC-`, `ORD-`, `REL-`, and `FAM-` are synthetic. Address strings use fictional cities, streets, buildings, units, and postcodes. Delete both files and run `make reproduce` to regenerate them.
