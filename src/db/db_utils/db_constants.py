PROPERTY_FIELDS = [
    'street',
    'y',
    'project',
    'x',
    'marketSegment',
]

TRANSACTION_FIELDS = [
    "area",
    "floorRange",
    "noOfUnits",
    "contractDate",
    "typeOfSale",
    "price",
    "propertyType",
    "district",
    "typeOfArea",
    "tenure",
]

INSERT_PROPERTY_QUERY = f"""
INSERT INTO Property ({','.join(PROPERTY_FIELDS)})
VALUES ({','.join(['%s']*len(PROPERTY_FIELDS))})
RETURNING id;
"""

INSERT_TRANSACTION_QUERY = f"""
INSERT INTO Transactions (property_id, {", ".join(TRANSACTION_FIELDS)}) 
VALUES ({{property_id}}, {','.join(['%s']*len(TRANSACTION_FIELDS))})
"""

LOAD_QUERY = """
    SELECT 
        p.id AS property_id,
        p.street,
        p.project,
        p.marketSegment,
        p.x,
        p.y,
        t.id AS transaction_id,
        t.area,
        t.floorRange,
        t.noOfUnits,
        t.contractDate,
        t.typeOfSale,
        t.price,
        t.propertyType,
        t.district,
        t.typeOfArea,
        t.tenure
    FROM 
        Property as p
    JOIN 
        Transactions as t
    ON 
        p.id = t.property_id;
"""
