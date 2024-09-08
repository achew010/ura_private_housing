CREATE TABLE Property (
    id SERIAL PRIMARY KEY,
    street TEXT,
    project TEXT,
    x TEXT,
    marketSegment TEXT,
    y TEXT
);

CREATE TABLE Transactions (
    id SERIAL PRIMARY KEY,
    property_id INT,
    area TEXT,
    floorRange TEXT,
    noOfUnits INTEGER, -- Assuming this is numeric
    contractDate TEXT,
    typeOfSale TEXT,
    price INTEGER, -- Assuming this is a numeric value
    propertyType TEXT,
    district TEXT,
    typeOfArea TEXT,
    tenure TEXT
);
