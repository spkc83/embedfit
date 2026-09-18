# Check Fraud / Deposit Risk Reference Use Case

This file provides domain context for examples and tests. It must not dictate generic library semantics.

## Potential labels

Illustrative only:

- legitimate
- duplicate presentment
- counterfeit
- altered amount
- altered payee
- washed
- forged signature
- synthetic
- other suspicious/fraud

## Potential relationship fields

- physical item ID
- check-template ID
- source bank
- depositing account
- suspected fraud-ring ID
- RDI event ID
- deposit channel

These can be used for grouping or sampling, but the library must treat them as arbitrary relationship fields.

## Why high resolution matters

Relevant visual evidence can be small:

- handwriting strokes
- MICR region
- amount-box edits
- payee edits
- signature
- scanner/compression artifacts
- repeated background/check stock

Therefore do not assume a full check should always be compressed to 224×224.

## Suggested future representation

```text
full image embedding
+ payee crop embedding
+ amount crop embedding
+ signature crop embedding
+ MICR crop embedding
→ learned or simple fusion
→ final item embedding
```

Automatic field detection is outside v0.1.

## RDI terminology

Institution-specific definitions and operational labels vary. Do not encode RDI event semantics into generic library code.

## Decisioning warning

A visual score is one feature in a broader risk process. Similarity to prior fraud is a signal, not proof of fraud.
