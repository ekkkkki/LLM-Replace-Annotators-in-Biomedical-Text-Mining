init_step = """
# TASK
This is a named entity recognization(NER) task in biomedical domain. Given a reseach article, your objective is to find all the "Disease" entities in the text.

# INSTRUCTION
- Please list up all the potential **mentions** along with a short explanation of your annotation.
- Also Annotate disease mentions that are used as Modifiers for otherconcepts
- Annotate minimum necessary span of text.
- Do not annotate organism names.
- Do not annotate gender.
- Do not annotate references to biological processes.
"""

structuralize_step = """
This is a structuralization task where you are required to structurilize an output from another LLM agent.
Based on the results from annother LLM agent, construct a structured HTML output to identify "ALL THE MENTIONS" of annotated entities in the "Input Article", and wrap them in HTML tags.

Instruction:
- Annotate all mentions (including duplicates, abbreviation, and alias) of annotated entities. 
- For entities with abbreviations, the full name and its abbreviation should be annotated separately(eg. "<category="Disease"><category="Disease">ataxia-telangiectasia</category> ( <category="Disease">A-T</category> )").

Example Input Article: ```Tumor suppression and apoptosis of human prostate carcinoma(PC) mediated by a genetic locus within human chromosome 10pter-q11.```
Example ChatGPT results: ```"Tumor suppression":Reasoning: Refers to mechanisms inhibiting tumor development, directly tied to disease processes like cancer. For prostate carcinoma, the word explicitly identifies "prostate carcinoma," a well-defined specific disease (prostate cancer).```
Example Output: ```<category="Disease" reason="Refers to mechanisms inhibiting tumor">Tumor</category> suppression and apoptosis of human <category="Disease" reason="a well-defined specific disease (prostate cancer)">prostate carcinoma</category>(<category="Disease" reason="a well-defined specific disease (prostate cancer)">PC</category>) mediated by a genetic locus within human chromosome 10pter-q11.```
Input Article: ```{text}```
ChatGPT results: ```{results}```
Structured output: \n
"""

validate_step = """
Your are an expert annotator in biomedical text mining. Your task is to check the accuracy of another annotator's annotations based on the annotation guideline:
Instruction: The annotation made by the other annotator will be shown in a HTML format. Check it and correct it if necessary. Your output should be in the same HTML format. No changes to the original text are allowed. 

# Annotation Guideline:
```
{annotation_guideline}
```
# Example Input: ```Tumor suppression and apoptosis of human <category="Disease">prostate carcinoma</category>(<category="Disease">PC</category>) mediated by a genetic locus within human chromosome 10pter-q11.```
# Example Corrected Output: ```<category="Disease" reason="Refers to mechanisms inhibiting tumor">Tumor</category> suppression and apoptosis of human <category="Disease" reason="a well-defined specific disease (prostate cancer)">prostate carcinoma</category>(<category="Disease" reason="a well-defined specific disease (prostate cancer)">PC</category>) mediated by a genetic locus within human chromosome 10pter-q11.```
# Input: ```{text}```
# Corrected output: \n
"""

annotation_guideline = """
# What to Annotate

## Annotate all Specific Disease mentions
A textual string referring to a disease name may refer to a **Specific Disease**, or a **Disease Class**. Disease mentions that could be described as a family of many Specific Diseases are annotated with an annotation category called **Disease Class**. The annotation category **Specific Disease** is used for mentions that could be linked to one specific definition that does not include further categorization.

**Example**:  
- Annotate: "Diastrophic dysplasia" as **Specific Disease** category and "autosomal recessive disease" as **Disease Class** category.  

## Annotate contiguous text strings
A textual string may refer to two or more separate disease mentions. Such mentions are annotated with the **Composite Mention** category.

**Example**:  
- The text phrase "Duchenne and Becker muscular dystrophy" refers to two separate diseases. If this phrase is separated into two strings: "Duchenne" and "Becker muscular dystrophy," it results in information loss because the word "Duchenne" on its own is not a disease mention.

## Annotate disease mentions that are used as Modifiers for other concepts
A textual string may refer to a disease name but it may modify a noun phrase, or not be a noun phrase. This is better expressed with the **Modifier** annotation category.

**Example**:  
- Annotate: "colorectal cancer" as **Modifier** category and "HNPCC" as **Modifier** category.

## Annotate duplicate mentions
For each sentence in the PubMed abstract and title, the locations of all disease mentions are marked, including duplicates within the same sentence.

## Annotate minimum necessary span of text
The minimum span of text necessary to include all the tokens expressing the most specific form of the disease is preferred.

**Example**:  
- In the case of "insulin-dependent diabetes mellitus", the disease mention including the whole phrase is preferred over its substrings such as "diabetes mellitus" or "diabetes".

## Annotate all synonymous mentions
Abbreviation definitions such as "Huntington disease" ("HD") are separated into two annotated mentions.

---

# What NOT to Annotate

## Do not annotate organism names
Organism names such as "human" are excluded from the preferred mention. Viruses, bacteria, and other organism names are not annotated unless it is clear from the context that the disease caused by these organisms is discussed.

**Example**:  
- "Epstein-Barr virus" and "cytomegalovirus" are annotated as **Specific Disease** category.

## Do not annotate gender
Tokens such as "male" and "female" are only included if they specifically identify a new form of the disease.

**Example**:  
- "male breast cancer"

## Do not annotate overlapping mentions
For example, the phrase "von Hippel-Lindau (VHL) disease" is annotated as one single disease mention, **Specific Disease** category.

## Do not annotate general terms
Very general terms such as "disease", "syndrome", "deficiency", "complications", "abnormalities", etc., are excluded. However, the terms "cancer" and "tumor" are retained.

## Do not annotate references to biological processes
For example, terms corresponding to biological processes such as "tumorigenesis" or "cancerogenesis".

## Do not annotate disease mentions interrupted by nested mentions
Basically, do not break the contiguous text rule.

**Example**:  
- "WT1 dysfunction is implicated in both neoplastic (Wilms tumor, mesothelioma, leukemia, and breast cancer) and nonneoplastic (glomerulosclerosis) disease."

In this example, the list of all disease mentions includes: "neoplastic disease" and "nonneoplastic disease" in addition to "Wilms tumor", "mesothelioma", "leukemia", "breast cancer", and "glomerulosclerosis". However, they were not annotated in our corpus because other tokens break up the phrases.

---

# Examples

- "Insulin-dependent diabetes mellitus", **Specific Disease**, prefer the whole string.
- "CDH1 mutations predispose to early onset colorectal cancer." "early onset" may or may not be part of the disease mention, depending on:
  1. If there is a UMLS concept specifying this as a separate form of disease, and
  2. If the annotator believes it should be included.
- Human "X-linked recessive disorder."
- "Huntington disease" ("HD"), the long form and the short form constitute two separate mentions.
- "C7 deficiency" associated with infectious complications. "infectious complications" is too general a term; **C7 deficiency** is a **Specific Disease**.
- "colorectal, endometrial, and ovarian cancers" is considered one **Composite Mention** of several **Specific Diseases**.
- "WT1 dysfunction is implicated in both neoplastic (Wilms tumor, mesothelioma, leukemias, and breast cancer) and nonneoplastic (glomerulosclerosis) disease."

---

# How Annotators Used the Annotation Tool

For each annotator:

1. Go to the annotation webpage.
2. Follow instructions to select your domain.
3. Work one annotation set at a time.
4. For each PubMed abstract:
   - Select the PMID to open the editor page. The editor page provides the following information:
     - The PMID links to the PubMed record.
     - Inside the editor, there are three fields: PMID, TITLE, and ABSTRACT.
     - Above the editor window are listed the annotation categories: **Specific Disease** (yellow), **Disease Class** (green), **Composite Mention** (blue), or **Modifier** (purple).
5. At the end of the editor is the "Submit" button.
6. Please annotate the title and abstract portion of the text.

### Annotating a New Mention
1. Select the string to be annotated by holding the mouse down and highlighting the whole selection.
2. Click on the appropriate category label listed above the editor window.

### Deleting a Previously Annotated Mention
1. Select the string to be removed from the annotations list by holding the mouse down and highlighting the whole selection.
2. Click on "Clear" label above the editor window.

### Final Steps
- After the abstract has been processed, press the "Submit" button found just below the editor window.
- To retrieve the last saved version of your annotations, click on the "Last Saved" button.
- To undo all your annotations about this particular document and start anew with the pre-annotated version, click on "Clear ALL, start from the beginning".
- The pre-annotated version considers all mentions as **Specific Disease** category. To change the category, hold the mouse button to highlight the text and click on the appropriate category label above the editor window.
- To change the span of a previously annotated mention, hold the mouse button to highlight the desired text and click on the appropriate category label above the editor window.

"""