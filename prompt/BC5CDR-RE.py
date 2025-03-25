prompt = """
This task involves Chemical-Induced Disease (CID) relation extraction. You will receive a research article containing information about various chemicals and diseases, often describing associations or the effects of chemicals on specific diseases. Additionally, you'll be provided with a list of normalized chemical and disease entities for reference.
Your goal is to identify all relevant (chemical, disease) pairs from the text, along with the context that indicates the chemical is associated with the disease, implying a chemical-induced disease relationship.

Your output should in a format of [(chemical_1, disease_1),(chemical_2, disease_2), ...]. For example, [(alfentanil,rigidity),(edrophonium, bradycardias)].

# Anotation Guideline for your reference:```
The chemical-induced disease relation pairs are annotated as part of the Comparative Toxicogenomics Database (CTD) curation. For the CDR task, some additional updates are performed such that:   
1. The annotated relationship includes primarily mechanistic relationships between a chemical and disease. Occasional biomarker relations are also included (e.g. relation between D006719 (Homovanillic Acid) and D006816 (Huntington Disease) in PMID:6453208).   
2. The relation should be explicitly mentioned in the abstract.   
3. Use the most specific disease in a relationship pair. 
```

# EXAMPLE 1
# EXAMPLE INPUT TEXT:
# EXAMPLE ENTITY LIST:
# EXAMPLE OUTPUT:

# INPUT TEXT:
# ENTITY LIST:
# OUTPUT:
"""