init_step = """
# TASK
This is a named entity recognization(NER) task in biomedical domain. Given a reseach article, your objective is to find all the "Chemical" entities and all the "Disease" entities in the text.

# INSTRUCTION
Please list up all the potential **mentions** along with a short explanation of your annotation.
"""

# In this case, we annotated disease/chemical entities separately
init_step_with_guideline = """
# TASK
This is a named entity recognization(NER) task in biomedical domain. Given a reseach article, your objective is to find all the "Disease" entities in the text.

# ANNOTATION REQUIREMENTS
{instruction_disease_or_chemical}
# INSTRUCTION
Please list up all the potential **mentions** along with a short explanation of your annotation.
"""

structuralize_step = """
This is a structuralization task where you are required to structurilize an output from another LLM agent.
Based on the results from annother LLM agent, construct a structured HTML output to identify "ALL THE MENTIONS" of annotated entities in the "Input Article", and wrap them in HTML tags.

# Instruction:
- Annotate all mentions (including duplicates, abbreviation, and alias) of annotated entities. 
- For entities with abbreviations, the full name and its abbreviation should be annotated separately(eg. "<category="Disease"><category="Disease">ataxia-telangiectasia</category> ( <category="Disease">A-T</category> )").

# (OPTIONAL) ANNOTATION GUIDELINE
{annotation_guideline}

# Example Input Article: ```Tumor suppression and apoptosis of human prostate carcinoma(PC) mediated by a genetic locus within human chromosome 10pter-q11.```
# Example ChatGPT results: ```"Tumor suppression":Reasoning: Refers to mechanisms inhibiting tumor development, directly tied to disease processes like cancer. For prostate carcinoma, the word explicitly identifies "prostate carcinoma," a well-defined specific disease (prostate cancer).```
# Example Output: ```<category="Disease" reason="Refers to mechanisms inhibiting tumor">Tumor</category> suppression and apoptosis of human <category="Disease" reason="a well-defined specific disease (prostate cancer)">prostate carcinoma</category>(<category="Disease" reason="a well-defined specific disease (prostate cancer)">PC</category>) mediated by a genetic locus within human chromosome 10pter-q11.```
# Input Article: ```{text}```
# ChatGPT results: ```{results}```
# Structured output: \n
"""

validate_step = """
Your are an expert annotator in biomedical text mining. Your task is to check the accuracy of another annotator's annotations based on the annotation guideline:
Instruction: The annotation made by the other annotator will be shown in a HTML format. Check it and correct it if necessary. Your output should be in the same HTML format. No changes to the original text are allowed. 

Annotation Guideline:
```
{annotation_guideline}
```
Example Input: ```Tumor suppression and apoptosis of human <category="Disease">prostate carcinoma</category>(<category="Disease">PC</category>) mediated by a genetic locus within human chromosome 10pter-q11.```
Example Corrected Output: ```<category="Disease" reason="Refers to mechanisms inhibiting tumor">Tumor</category> suppression and apoptosis of human <category="Disease" reason="a well-defined specific disease (prostate cancer)">prostate carcinoma</category>(<category="Disease" reason="a well-defined specific disease (prostate cancer)">PC</category>) mediated by a genetic locus within human chromosome 10pter-q11.```
Input: ```{text}```
Corrected output: \n
"""

# manually summarized from annotation guideline used for placeholder in init_step
# for disease
instruction_disease = """
## What to Annotate
1. Specific Disease Mentions(Prefer the most specific term available).
2. Minimal Text Span(Annotate only the necessary words)
3. All Occurrences
4. Morphological Variants
5. Annotate both full name and abbreviation separately

## What NOT to Annotate
1. Species Names
2. Overlapping Mentions: Select the broader disease when overlaps occur (e.g., "hepatitis B virus infected" → only "Hepatitis B" annotated).
3. General Terms(terms like "disease," "syndrome," and "complications" (except specific terms like "cancer" or "tumor").)
4. Biological Processes: terms referring to disease processes (e.g., "tumorigenesis" or "cancerogenesis").
"""

# for chemical
instruction_chemical = """
## What to Annotate
1. Annotate the most specific chemical mentions. Chemicals which should be annotated are listed as follows: 
   a) Chemical Nouns convertible to a single chemical structure diagram, a general Markush diagram with R groups.
   b) General class names where the definition of the class includes information on some structural or elemental composition
   c) Small Biochemicals 
   d) Synthetic Polymers
   e) Special chemicals having well-defined chemical compositions. 
 
2. Annotate all mentions of a chemical entity in an abstract. 
3. Annotate abbreviations. 
    
## What not to annotate?    
1. DO NOT annotate other terms different from chemical nouns. 
2. DO NOT annotate chemical nouns named for a role or similar, that is, nonstructural concepts.
3. DO NOT annotate very nonspecific structural concepts. e.g. Atom, Ion, Protein ... 
4. DO NOT annotate words that are not chemicals in context, even if they are co-incidentally the same set of characters (synonyms and metaphors). 
5. DO NOT annotate Biomolecules/Macromolecular biochemicals
6. DO NOT annotate general vague compositions. 
7. DO NOT annotate special words not to be labeled by convention (e.g. Water)
"""

full_annotation_guideline_disease = """
## What to Annotate

1. **Most Specific Disease Mentions:**  
   Annotate the most specific disease mentions and select the best-matching MeSH ID.  
   - *Example:* Prefer "partial seizures" over "seizures."

2. **Minimum Necessary Text Spans:**  
   Annotate only the minimum necessary span for disease mentions.  
   - *Example:* Use "hypertension" instead of "sustained hypertension."

3. **All Mentions Including Duplicates:**  
   Annotate all mentions of a disease entity within an abstract, including duplicates in the same sentence. PubTator has an automatic duplicate-marking function.

4. **Morphological Variations:**  
   Annotate disease mentions with morphological variations, such as adjectives.  
   - *Example:* Annotate "hypertensive" as "hypertension."

5. **Abbreviations:**  
   Annotate abbreviations separately.  
   - *Example:* Annotate "Huntington disease (HD)" as two annotations: "Huntington disease" and "HD," both linked to the same concept ID.

6. **Composite Disease Mentions:**  
   Annotate all concepts in composite disease mentions using the `|` separator. When possible, spell out individual constituents with associated text.  
   - *Example:* "ovarian and peritoneal cancer" → D010051 (Ovarian Neoplasms) | D010534 (Peritoneal Neoplasms); also provide "ovarian ... cancer" (D010051) and "peritoneal cancer" (D010534).

7. **Multiple Concepts for One Disease Mention:**  
   Use the `+` concatenator when a disease mention logically requires multiple concepts. Spell out individual constituents when possible.  
   - *Example:* "bone marrow oedema" → D001855 (Bone Marrow Diseases) + D004487 (Edema); also provide "bone marrow" (D001855) and "edema" (D004487).

8. **Unnormalizable Diseases:**  
   Use "-1" for diseases that cannot be normalized.  
   - *Example:* "erythroblastocytopenia" → "-1."

---

## What Not to Annotate

1. **Species Names:**  
   Exclude organism names unless essential to the disease name or explicitly indicated as disease-causing.  
   - *Example:* Annotate "HIV" in "HIV-1-infected," but exclude "human."

2. **Overlapping Mentions:**  
   Avoid annotating overlapping mentions.  
   - *Example:* Annotate "hepatitis B virus (HBV) infected" as a single disease (D006509, Hepatitis B).

3. **General Terms:**  
   Exclude general terms like "disease," "syndrome," "deficiency," "complications," but retain terms such as "pain," "cancer," "tumor," and "death."

4. **Biological Processes:**  
   Do not annotate biological process references such as "tumorigenesis" or "cancerogenesis."

---

## Special Cases

### Toxicity Mentions
- **General Toxicity:** Use D064420 (Drug-Related Side Effects and Adverse Reactions).
- **Specific Toxicity Types:** Use specific concepts under D064420 when available.  
  - *Examples:* "cardiotoxicity" → D066126; "liver toxicity" → D056486.
- **No Matching MeSH:** Use corresponding disease IDs if no match under D064420.  
  - *Examples:* "visual toxicity" → D014786 (vision disorders); "auditory toxicity"/"ototoxicity" → D006311 (hearing disorders).

### Drug-Induced Diseases
- Annotate drug-induced diseases using concepts under D064420 whenever possible.  
  - *Examples:*  
    - "dyskinesia" (drug-induced context) → D004409 instead of its original ID (D020820).  
    - "hepatitis" (drug-induced) → D056486 (Drug-induced Liver Injury).  
    - "akathisia" (drug-induced) → D017109 (Akathisia, Drug-induced).
"""

full_annotation_guideline_chemical = """
```markdown
## What to Annotate

1. **Most Specific Chemical Mentions:**  
   Annotate the most specific chemical mentions and select the best-matching MeSH concept ID. Chemicals to annotate include:

   - **Chemical Nouns:** Convertible to single chemical structures such as single atoms, ions, isotopes, pure elements, and molecules (e.g., Calcium [Ca], Iron [Fe], Lithium [Li], Potassium [K], Oxygen [O₂]).
   - **General Class Names:** Structural or elemental composition defined classes (e.g., steroids, sugars, fatty acids, saturated fatty acids).
   - **Small Biochemicals:** 
     - Monosaccharides, disaccharides, trisaccharides (e.g., glucose, sucrose).
     - Peptides and proteins (<15 amino acids, e.g., Angiotensin II).
     - Monomers, dimers, trimers of nucleotides (e.g., ATP, cAMP).
     - Fatty acids and derivatives, excluding polymeric structures (e.g., cholesterol, glycerol, prostaglandin E₁).
   - **Synthetic Polymers:** (e.g., polyethylene glycol).
   - **Special Chemicals:** Well-defined chemical compositions (e.g., "ethanolic extract of Daucus carota seeds (DCE)", "grape seed proanthocyanidin extract").

2. **All Chemical Mentions:**  
   Annotate every chemical entity mention within an abstract.

3. **Abbreviations:**  
   Annotate abbreviations, carefully using context to avoid ambiguity (e.g., Nitric Oxide [NO]).

4. **Unnormalizable Chemicals:**  
   Annotate with concept ID "-1" if chemicals cannot be normalized.

---

## What Not to Annotate

1. **Non-chemical Nouns and Adjectives:**  
   Exclude adjectives derived from chemical names (e.g., muscarinic, adrenergic, purinergic).

2. **Role-based Chemical Nouns:**  
   Do not annotate chemicals defined by role rather than structure (e.g., anti-HIV agents, anticonvulsants, antipsychotic, anticoagulant).

3. **Nonspecific Structural Concepts:**  
   Exclude very general terms (e.g., atom, ion, molecular, lipid, protein).

4. **Contextually Non-chemical Words:**  
   Exclude words coincidentally identical to chemicals if contextually non-chemical (e.g., "gold" in "gold standard").

5. **Biomolecules/Macromolecules:**  
   Exclude large biomolecules and established DNA/RNA/protein sequences (e.g., insulin, DNA, mRNA, collagen, starch, cellulose, glycogen, glucocorticoid, glucagon [29 peptides], prolactin [199 peptides]).

6. **General Vague Compositions:**  
   Exclude broad or vague chemical compositions (e.g., opiate).

7. **Conventionally Non-annotated Terms:**  
   Exclude conventionally non-annotated substances (e.g., water, saline, juice).

---

## Special Cases

- **Special Classes:**  
  Annotate specific classes to match CTD’s relation annotations, such as Antidepressive Agents (D000928), Estrogens (D004967), and Oral Contraceptives (D003276).

- **Combination Drugs:**  
  Annotate combo drugs as one entity, using the corresponding MeSH ID where available, instead of separate components.  
  - *Example:* Annotate "levodopa/carbidopa" with MeSH ID: C009265.
```

"""