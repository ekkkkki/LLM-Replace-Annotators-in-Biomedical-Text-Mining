# do all steps at once (w/ wo/ dynamic few-shot)
basic_prompt = """
# TASK
In this task, you will engage in relation extraction from a biomedical research article. You will be given a research article and a list of entities. Your objective is to identify and categorize relationships between those biomedical entities in the text. Each identified relationship should be expressed in tuples such as [(entity_1, entity_2, Relationship Type), (entity_3, entity_4, Relationship Type), ...].
Remember that there may be mutiple mentions(including alias, abbreviation, etc) for each entity in the text, and you should consider all of them when identifying relationships.

# Type of Entities:
Gene/Gene Product: Include entities like paralogs and orthologs (e.g., bone morphogenetic protein (bmp)). Avoid annotating function-based families like tumor suppressor genes or protein kinases.
Chemical: Annotate chemicals including dosage information (e.g., GFC75, where 75 is the dose). Exclude antibodies, non-active substances like saline or placebo, and broad categories like vaccines.
Disease: Focus on specific diseases and phenotypic features excluding high-level categories like genetic disorders. Include specific conditions like drug addiction and relevant symptoms.
Variant: Include genomic or protein variants following the tmVar guidelines. Avoid annotating gene modifications not related to specific variants.
Species: Annotate at the species level, excluding higher taxonomic classifications like mammalian.
Cell Line: Identify mentions of specific cell lines used in the research.

# Types of Relationships:
Positive Correlation: Indicates a direct, positive linkage between two entities where one influences the other beneficially.
Negative Correlation: Indicates an inverse relationship where the presence or increase of one entity negatively impacts the other.
Association: Describes a general, possibly indirect linkage or co-occurrence between two entities without implying a direct cause-effect relationship.
Output Format:
Your output should list all identified relationships in the format mentioned, like [(betaine, myocardial ischemic injury, Negative_Correlation), (polysaccharide, MHC protein complex, Association)].

# Instructions:
We only annotate the following associations between the following entities pairs:
'Chemical-Chemical', 'Chemical-Disease', 'Chemical-Gene/Gene Product', 'Disease-Disease', 'Disease-Gene/Gene Product', 'Gene/Gene Product-Gene/Gene Product'.
Please ignore any other associations that do not fall into these categories.

# Example Input Entities: 
# Example Input Article: 
# Example Output:
# Input Entity Pairs:
# Input Article:
# Output:
"""

# proposed pipeline utilizing annotation guideline
# step 1: predict the exist of relation between candidate entities
step_1 = """
# TASK
Given a research article and a list of entity pairs, your task is to identify whether there is a potential relationship between the two entities for each entity pair.

# Instruction
Output a list of triplet where each triplet include a entity pair and the result (True/Flase) like following: [(entity_1, entity_2, True), (entity_3, entity_4, Flase),...]
Remember that if there is no relationship **strictly meet the following introduction**, please just output "False".

# Guideline for your Annotation
{entity_pair_guideline}

# Input Entity Pairs:
# Input Article:
# Output:
"""

# step 2: predict the type of relation between entity pairs
step_2 = """
# TASK
Given a research article and a list of entity pairs, your task is to identify the type of relationship between the two entities for each entity pair.

# Instruction
Output a list of triplet where each triplet include a entity pair and the relationship type like following: [(entity_1, entity_2, Relationship Type), (entity_3, entity_4, Relationship Type),...]
Type of Relationships can be found in the following annotation guideline. However, if there is no relationship **strictly meet the following introduction**, please just output "False" for the relationship type.

# Guideline for your Annotation
{relation_type_guideline}

# Input Entity Pairs:
# Input Article:
# Output:
"""

# step 3: validation
step_3 = """
# Task
Your are an expert annotator in biomedical text mining. Your task is to check the accuracy of another annotator's annotations in a relation extraction task based on the annotation guideline:

# Instruction
The annotation made by the other annotator will be shown in a list of tuplets. Check it and correct it if necessary. Your output should be in the same format.

# Annotation Guideline:
```
{annotation_guideline_for_validation}
```
# Example Input: ```
    [
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Creatinine", "Positive_Correlation"),
        ("Renal dysfunction (nephrotoxic)", "Cyclosporine", "Negative_Correlation"),
        ("Renal dysfunction (nephrotoxic)", "ESRD", "Negative_Correlation"),
        ("Renal dysfunction (nephrotoxic)", "Tacrolimus", "Positive_Correlation"),
        ("Hepatorenal syndrome", "Creatinine", "Positive_Correlation")
    ]
```

# Example Input Text: End-stage renal disease (ESRD) after orthotopic liver transplantation (OLTX) using calcineurin-based immunotherapy: risk of development and treatment. BACKGROUND: The calcineurin inhibitors cyclosporine and tacrolimus are both known to be nephrotoxic. Their use in orthotopic liver transplantation (OLTX) has dramatically improved success rates. Recently, however, we have had an increase of patients who are presenting after OLTX with end-stage renal disease (ESRD). This retrospective study examines the incidence and treatment of ESRD and chronic renal failure (CRF) in OLTX patients. METHODS: Patients receiving an OLTX only from June 1985 through December of 1994 who survived 6 months postoperatively were studied (n=834). Our prospectively collected database was the source of information. Patients were divided into three groups: Controls, no CRF or ESRD, n=748; CRF, sustained serum creatinine >2.5 mg/dl, n=41; and ESRD, n=45. Groups were compared for preoperative laboratory variables, diagnosis, postoperative variables, survival, type of ESRD therapy, and survival from onset of ESRD. RESULTS: At 13 years after OLTX, the incidence of severe renal dysfunction was 18.1% (CRF 8.6% and ESRD 9.5%). Compared with control patients, CRF and ESRD patients had higher preoperative serum creatinine levels, a greater percentage of patients with hepatorenal syndrome, higher percentage requirement for dialysis in the first 3 months postoperatively, and a higher 1-year serum creatinine. Multivariate stepwise logistic regression analysis using preoperative and postoperative variables identified that an increase of serum creatinine compared with average at 1 year, 3 months, and 4 weeks postoperatively were independent risk factors for the development of CRF or ESRD with odds ratios of 2.6, 2.2, and 1.6, respectively. Overall survival from the time of OLTX was not significantly different among groups, but by year 13, the survival of the patients who had ESRD was only 28.2% compared with 54.6% in the control group. Patients developing ESRD had a 6-year survival after onset of ESRD of 27% for the patients receiving hemodialysis versus 71.4% for the patients developing ESRD who subsequently received kidney transplants. CONCLUSIONS: Patients who are more than 10 years post-OLTX have CRF and ESRD at a high rate. The development of ESRD decreases survival, particularly in those patients treated with dialysis only. Patients who develop ESRD have a higher preoperative and 1-year serum creatinine and are more likely to have hepatorenal syndrome. However, an increase of serum creatinine at various times postoperatively is more predictive of the development of CRF or ESRD. New strategies for long-term immunosuppression may be needed to decrease this complication.# Example Corrected Output: ```
# Example Corrected Output: ```
    [
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Creatinine", "Positive_Correlation"),
        ("Renal dysfunction (nephrotoxic)", "Cyclosporine", "Positive_Correlation"),
        ("Renal dysfunction (nephrotoxic)", "Tacrolimus", "Positive_Correlation"),
        ("Hepatorenal syndrome", "Creatinine", "Positive_Correlation")
    ]
```

# Input: ```{text}```
Corrected output: \n
"""

# ----------------------------
# Annotation guideline snippets used for {entity_pair_guideline} placeholder in step_1
# One fixed example is presented in prompt for demonstration, while dynamic few-shot examples can also be applied here.
# ----------------------------
disease_chemical_pair = """
    * Co-occurrence in the same sentence is not required.
    * DO NOT annotate a confirmed irrelevance. 
    * Annotate the associations with explicit statements in the text. 
    * The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear.
    * Annotate the association of chemicals and disease due to the “toxicity” and “drug-induced”, but not annotate the chemicals with the disease concept-“toxicity” directly.
    * Annotate the relation between the target disease and the measure level of the molecules/chemicals in blood test with correlation.
    * DO NOT annotate the association of disease to the staining chemicals.
    * DO NOT annotate the association of any disease with the chemical concept “analgesia” in “patient-controlled analgesia (PCA) pump”.
    * DO NOT annotate toxicity with its chemical.
    * Annotate the chemicals which prevents the chemical-induced disease.

    # Example 1:
    # Example Input Entity pairs: ```
    [
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Cyclosporine"),
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Tacrolimus"),
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Creatinine"),
        ("Renal dysfunction (nephrotoxic)", "Cyclosporine"),
        ("Renal dysfunction (nephrotoxic)", "Tacrolimus"),
        ("Renal dysfunction (nephrotoxic)", "Creatinine"),
        ("Hepatorenal syndrome", "Cyclosporine"),
        ("Hepatorenal syndrome", "Tacrolimus"),
        ("Hepatorenal syndrome", "Creatinine")
    ]```
    # Example Input Text: 
    End-stage renal disease (ESRD) after orthotopic liver transplantation (OLTX) using calcineurin-based immunotherapy: risk of development and treatment. BACKGROUND: The calcineurin inhibitors cyclosporine and tacrolimus are both known to be nephrotoxic. Their use in orthotopic liver transplantation (OLTX) has dramatically improved success rates. Recently, however, we have had an increase of patients who are presenting after OLTX with end-stage renal disease (ESRD). This retrospective study examines the incidence and treatment of ESRD and chronic renal failure (CRF) in OLTX patients. METHODS: Patients receiving an OLTX only from June 1985 through December of 1994 who survived 6 months postoperatively were studied (n=834). Our prospectively collected database was the source of information. Patients were divided into three groups: Controls, no CRF or ESRD, n=748; CRF, sustained serum creatinine >2.5 mg/dl, n=41; and ESRD, n=45. Groups were compared for preoperative laboratory variables, diagnosis, postoperative variables, survival, type of ESRD therapy, and survival from onset of ESRD. RESULTS: At 13 years after OLTX, the incidence of severe renal dysfunction was 18.1% (CRF 8.6% and ESRD 9.5%). Compared with control patients, CRF and ESRD patients had higher preoperative serum creatinine levels, a greater percentage of patients with hepatorenal syndrome, higher percentage requirement for dialysis in the first 3 months postoperatively, and a higher 1-year serum creatinine. Multivariate stepwise logistic regression analysis using preoperative and postoperative variables identified that an increase of serum creatinine compared with average at 1 year, 3 months, and 4 weeks postoperatively were independent risk factors for the development of CRF or ESRD with odds ratios of 2.6, 2.2, and 1.6, respectively. Overall survival from the time of OLTX was not significantly different among groups, but by year 13, the survival of the patients who had ESRD was only 28.2% compared with 54.6% in the control group. Patients developing ESRD had a 6-year survival after onset of ESRD of 27% for the patients receiving hemodialysis versus 71.4% for the patients developing ESRD who subsequently received kidney transplants. CONCLUSIONS: Patients who are more than 10 years post-OLTX have CRF and ESRD at a high rate. The development of ESRD decreases survival, particularly in those patients treated with dialysis only. Patients who develop ESRD have a higher preoperative and 1-year serum creatinine and are more likely to have hepatorenal syndrome. However, an increase of serum creatinine at various times postoperatively is more predictive of the development of CRF or ESRD. New strategies for long-term immunosuppression may be needed to decrease this complication.
    # Example Output: ```
    [
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Cyclosporine", False),
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Tacrolimus", False),
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Creatinine", True),
        ("Renal dysfunction (nephrotoxic)", "Cyclosporine", True),
        ("Renal dysfunction (nephrotoxic)", "Tacrolimus", True),
        ("Renal dysfunction (nephrotoxic)", "Creatinine", False),
        ("Hepatorenal syndrome", "Cyclosporine", False),
        ("Hepatorenal syndrome", "Tacrolimus", False),
        ("Hepatorenal syndrome", "Creatinine", True)
    ]```
"""

disease_gene_pair = """
    * Co-occurrence in the same sentence is not required.
    * DO NOT annotate a confirmed irrelevance. 
    * Annotate the associations with explicit statements in the text. 
    * The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear.
    * DO NOT annotate the prognostic factor of genes or its variants to the target disease.
    * Annotate the negative correlation (e.g., knockdown gene to the target disease).
    * DO NOT annotate the symptoms of the target disease to the correlated variants or genes, unless the clear statement between the symptom and variant is provided.

    # Example 1:
    # Example Input Entities: ```
    [
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "HNF-6 (Hepatocyte nuclear factor-6)"),
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "Insulin"),
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "MODY genes (MODY1, MODY3, MODY4)")
    ]
    ```
    # Example Input Text: Hepatocyte nuclear factor-6: associations between genetic variability and type II diabetes and between genetic variability and estimates of insulin secretion. The transcription factor hepatocyte nuclear factor (HNF)-6 is an upstream regulator of several genes involved in the pathogenesis of maturity-onset diabetes of the young. We therefore tested the hypothesis that variability in the HNF-6 gene is associated with subsets of Type II (non-insulin-dependent) diabetes mellitus and estimates of insulin secretion in glucose tolerant subjects.   We cloned the coding region as well as the intron-exon boundaries of the HNF-6 gene. We then examined them on genomic DNA in six MODY probands without mutations in the MODY1, MODY3 and MODY4 genes and in 54 patients with late-onset Type II diabetes by combined single strand conformational polymorphism-heteroduplex analysis followed by direct sequencing of identified variants. An identified missense variant was examined in association studies and genotype-phenotype studies.   We identified two silent and one missense (Pro75 Ala) variant. In an association study the allelic frequency of the Pro75Ala polymorphism was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched glucose tolerant control subjects. Moreover, in studies of 238 middle-aged glucose tolerant subjects, of 226 glucose tolerant offspring of Type II diabetic patients and of 367 young healthy subjects, the carriers of the polymorphism did not differ from non-carriers in glucose induced serum insulin or C-peptide responses.   Mutations in the coding region of the HNF-6 gene are not associated with Type II diabetes or with changes in insulin responses to glucose among the Caucasians examined.
    # Example Output: 
        [
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "HNF-6 (Hepatocyte nuclear factor-6)", True),
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "Insulin", False),
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "MODY genes (MODY1, MODY3, MODY4)", False)
    ]
"""

disease_variant_pair = """
    * Co-occurrence in the same sentence is not required.
    * DO NOT annotate a confirmed irrelevance. 
    * Annotate the associations with explicit statements in the text. 
    * The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear.
    * Annotate the corresponding diseases with the variant when it is observed in patients.
    * Annotate the diseases with the associated genes.
    * Annotate the corresponding gene of the variant with the variant-associated disease.
    * Annotate the association between the variants (and corresponding gene) and the diseases belonging to the target disease.
    * Annotate the minor association (e.g., the observation of the association on few patients).
    * Annotate the associations in the particular races.
    * Annotate the association between disease and variant if the inheritance of the genetic disease is confirmed by the variant.
    * DO NOT annotate the symptoms of the target disease to the correlated variants or genes, unless the clear statement between the symptom and variant is provided.
    
    # Example 1:
    # Example Input Entities: ```
    [
    ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "5382insC"),
    ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "C61G (rs28897672)"),
    ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "4153delA"),
    ("cancer", "5382insC"),
    ("cancer", "C61G (rs28897672)"),
    ("cancer", "4153delA"),
    ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "5382insC"),
    ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "C61G (rs28897672)"),
    ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "4153delA"),
    ("ovarian cancer (ovarian cancers)", "5382insC"),
    ("ovarian cancer (ovarian cancers)", "C61G (rs28897672)"),
    ("ovarian cancer (ovarian cancers)", "4153delA"),
    ("BRCA1 abnormalities", "5382insC"),
    ("BRCA1 abnormalities", "C61G (rs28897672)"),
    ("BRCA1 abnormalities", "4153delA")
    ]

    ```
    # Example Input Text: Hepatocyte nuclear factor-6: associations between genetic variability and type II diabetes and between genetic variability and estimates of insulin secretion. The transcription factor hepatocyte nuclear factor (HNF)-6 is an upstream regulator of several genes involved in the pathogenesis of maturity-onset diabetes of the young. We therefore tested the hypothesis that variability in the HNF-6 gene is associated with subsets of Type II (non-insulin-dependent) diabetes mellitus and estimates of insulin secretion in glucose tolerant subjects.   We cloned the coding region as well as the intron-exon boundaries of the HNF-6 gene. We then examined them on genomic DNA in six MODY probands without mutations in the MODY1, MODY3 and MODY4 genes and in 54 patients with late-onset Type II diabetes by combined single strand conformational polymorphism-heteroduplex analysis followed by direct sequencing of identified variants. An identified missense variant was examined in association studies and genotype-phenotype studies.   We identified two silent and one missense (Pro75 Ala) variant. In an association study the allelic frequency of the Pro75Ala polymorphism was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched glucose tolerant control subjects. Moreover, in studies of 238 middle-aged glucose tolerant subjects, of 226 glucose tolerant offspring of Type II diabetic patients and of 367 young healthy subjects, the carriers of the polymorphism did not differ from non-carriers in glucose induced serum insulin or C-peptide responses.   Mutations in the coding region of the HNF-6 gene are not associated with Type II diabetes or with changes in insulin responses to glucose among the Caucasians examined.
    # Example Output: 
    [
    ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "5382insC", True),
    ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "C61G (rs28897672)", True),
    ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "4153delA", True),
    ("cancer", "5382insC", False),
    ("cancer", "C61G (rs28897672)", False),
    ("cancer", "4153delA", False),
    ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "5382insC", True),
    ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "C61G (rs28897672)", True),
    ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "4153delA", True),
    ("ovarian cancer (ovarian cancers)", "5382insC", True),
    ("ovarian cancer (ovarian cancers)", "C61G (rs28897672)", True),
    ("ovarian cancer (ovarian cancers)", "4153delA", True),
    ("BRCA1 abnormalities", "5382insC", True),
    ("BRCA1 abnormalities", "C61G (rs28897672)", True),
    ("BRCA1 abnormalities", "4153delA", True)
]

"""

gene_gene_pair = """
    * Co-occurrence in the same sentence is not required.
    * DO NOT annotate a confirmed irrelevance. 
    * Annotate the associations with explicit statements in the text. 
    * The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear.
    * Annotate the pairs of the members in the protein complex, e.g., EPO/EPOR (PMID:22808010).
    * DO NOT annotate the “is_a” association between two genes. For instance, DO NOT annotate ”breast and ovarian cancer gene” and “BRCA1”.
    * Annotate protein-protein interaction for singling proteins and the receptors.

    # Example 1:
    # Example Input Entities: ```
    [
        ("IL1B (interleukin 1 beta, IL1beta, Proinflammatory cytokine)", "NFkappaB (nuclear factor of kappa B)"),
        ("IL1B (interleukin 1 beta, IL1beta, Proinflammatory cytokine)", "FAS (FS-7 cell-associated cell surface antigen)"),
        ("NFkappaB (nuclear factor of kappa B)", "FAS (FS-7 cell-associated cell surface antigen)")
    ]
    # Example Input Text: A polymorphism of C-to-T substitution at -31 IL1B is associated with the risk of advanced gastric adenocarcinoma in a Japanese population. Proinflammatory cytokine gene polymorphisms have been demonstrated to associate with gastric cancer risk, of which IL1B-31T/C and -511C/T changes have been well investigated due to the possibility that they may alter the IL1B transcription. The signal transduction target upon interleukin 1 beta (IL1beta) stimulation, the nuclear factor of kappa B (NFkappaB) activation, supports cancer development, signal transduction in which is mediated by FS-7 cell-associated cell surface antigen (FAS) signaling. Based on recent papers describing the prognostic roles of the polymorphisms and the NFkappaB functions on cancer development, we sought to determine if Japanese gastric cancer patients were affected by the IL1B -31/-511 and FAS-670 polymorphisms. A case-control study was conducted on incident gastric adenocarcinoma patients (n=271) and age-gender frequency-matched control subjects (n=271). We observed strong linkage disequilibrium between the T allele at -511 and the C allele at -31 and between the C allele at -511 and the T allele at -31 in IL1B in both the cases and controls (R (2)=0.94). Neither IL1B-31, -511 nor FAS-670 polymorphisms showed significantly different risks of gastric adenocarcinoma. Though FAS-670 polymorphisms did not show any significant difference, the proportion of subjects with IL1B-31TT (or IL1B-511CC) increased according to stage (trend P=0.019). In particular, subjects with stage IV had a two times higher probability of having either IL1B-31TT (or IL1B-511CC) genotype compared with stage I subjects. These observations suggest that IL1B-31TT and IL1B-511CC are associated with disease progression.
    # Example Output: ```
    [
        ("IL1B (interleukin 1 beta, IL1beta, Proinflammatory cytokine)", "NFkappaB (nuclear factor of kappa B)", True),
        ("IL1B (interleukin 1 beta, IL1beta, Proinflammatory cytokine)", "FAS (FS-7 cell-associated cell surface antigen)", False),
        ("NFkappaB (nuclear factor of kappa B)", "FAS (FS-7 cell-associated cell surface antigen)", False)
    ]
    ```


"""

gene_chemical_pair = """
    * Co-occurrence in the same sentence is not required.
    * DO NOT annotate a confirmed irrelevance. 
    * Annotate the associations with explicit statements in the text. 
    * The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear.
    * Annotate the chemical with the gene receptor. For instance, "antipsychotic drugs that have a high affinity with the D2 receptor” 

    # Example 1:
    # Example Input Entities: ```
    [
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "Chloroacetaldehyde"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "sulfhydryl reagent"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "thiol"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "ifosfamide"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "alkylating agent"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "LDH"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "trypan blue"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "acrolein"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "cisplatin")
    ]
    ```
    # Example Input Text: Chloroacetaldehyde as a sulfhydryl reagent: the role of critical thiol groups in ifosfamide nephropathy. Chloroacetaldehyde (CAA) is a metabolite of the alkylating agent ifosfamide (IFO) and putatively responsible for renal damage following anti-tumor therapy with IFO. Depletion of sulfhydryl (SH) groups has been reported from cell culture, animal and clinical studies. In this work the effect of CAA on human proximal tubule cells in primary culture (hRPTEC) was investigated. Toxicity of CAA was determined by protein content, cell number, LDH release, trypan blue exclusion assay and caspase-3 activity. Free thiols were measured by the method of Ellman. CAA reduced hRPTEC cell number and protein, induced a loss in free intracellular thiols and an increase in necrosis markers. CAA but not acrolein inhibited the cysteine proteases caspase-3, caspase-8 and cathepsin B. Caspase activation by cisplatin was inhibited by CAA. In cells stained with fluorescent dyes targeting lysosomes, CAA induced an increase in lysosomal size and lysosomal leakage. The effects of CAA on cysteine protease activities and thiols could be reproduced in cell lysate. Acidification, which slowed the reaction of CAA with thiol donors, could also attenuate effects of CAA on necrosis markers, thiol depletion and cysteine protease inhibition in living cells. Thus, CAA directly reacts with cellular protein and non-protein thiols, mediating its toxicity on hRPTEC. This effect can be reduced by acidification. Therefore, urinary acidification could be an option to prevent IFO nephropathy in patients.
    # Example Output: ```
    [
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "Chloroacetaldehyde", True),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "sulfhydryl reagent", False),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "thiol", False),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "ifosfamide", False),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "alkylating agent", False),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "LDH", False),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "trypan blue", False),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "acrolein", False),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "cisplatin", True)
    ]    
    ```
"""

chemical_chemical_pair = """
    * Co-occurrence in the same sentence is not required.
    * DO NOT annotate a confirmed irrelevance. 
    * Annotate the associations with explicit statements in the text. 
    * The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear.
    * DO NOT annotate the “is_a” association between two chemicals. 
    * Annotate the chemical-chemical association for the chemical contributes to the other for treatment, side effects and drug-drug interaction. 

    # Example 1:
    # Example Input Entities: ```
    [
        ("phosphatidylethanolamine", "phosphatidylcholine")
    ]
    ```
    # Example Input Text: The phosphatidylethanolamine N-methyltransferase gene V175M single nucleotide polymorphism confers the susceptibility to NASH in Japanese population. BACKGROUND/AIMS: The genetic predisposition on the development of nonalcoholic steatohepatitis (NASH) has been poorly understood. A functional polymorphism Val175Met was reported in phosphatidylethanolamine N-methyltransferase (PEMT) that catalyzes the conversion of phosphatidylethanolamine to phosphatidylcholine. The aim of this study was to investigate whether the carriers of Val175Met variant impaired in PEMT activity are more susceptible to NASH. METHODS: Blood samples of 107 patients with biopsy-proven NASH and of 150 healthy volunteers were analyzed by the polymerase chain reaction (PCR) and restriction fragment length polymorphism. RESULTS: Val175Met variant allele of the PEMT gene was significantly more frequent in NASH patients than in healthy volunteers (p<0.001), and carriers of Val175Met variant were significantly more frequent in NASH patients than in healthy volunteers (p<0.01). Among NASH patients, body mass index was significantly lower (p<0.05), and non-obese patients were significantly more frequent (p<0.001) in carriers of Val175Met variant than in homozygotes of wild type PEMT. CONCLUSIONS: Val175Met variant of PEMT could be a candidate molecule that determines the susceptibility to NASH, because it is more frequently observed in NASH patients and non-obese persons with Val175Met variant of PEMT are facilitated to develop NASH.
    # Example Output: ```
    [
        ("phosphatidylethanolamine", "phosphatidylcholine", True)
    ]
    ```
"""

chemical_variant_pair = """
    * Co-occurrence in the same sentence is not required.
    * DO NOT annotate a confirmed irrelevance. 
    * Annotate the associations with explicit statements in the text. 
    * The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear.
    * Annotate the chemical with the variant, if the chemical binding ability of the gene impaired by the variant. 
    * Annotate the chemical with the variant, if different mRNA stability between different alleles affected by the chemical.

    # Example 1:
    # Example Input Entities: ```
    [
        ("glucose", "rs5966709"),
        ("glucose", "rs4828037"),
        ("glucose", "rs11798018"),
        ("glucose", "rs2073163"),
        ("glucose", "rs1155794"),
        ("glucose", "rs2073162"),
        ("glucose", "rs1155974")
    ]
    ```
    # Example Input Text: Tenomodulin is associated with obesity and diabetes risk: the Finnish diabetes prevention study. We recently showed that long-term weight reduction changes the gene expression profile of adipose tissue in overweight individuals with impaired glucose tolerance (IGT). One of the responding genes was X-chromosomal tenomodulin (TNMD), a putative angiogenesis inhibitor. Our aim was to study the associations of individual single nucleotide polymorphisms and haplotypes with adiposity, glucose metabolism, and the risk of type 2 diabetes (T2D). Seven single nucleotide polymorphisms from two different haploblocks were genotyped from 507 participants of the Finnish Diabetes Prevention Study (DPS). Sex-specific genotype effects were observed. Three markers of haploblock 1 were associated with features of adiposity in women (rs5966709, rs4828037) and men (rs11798018). Markers rs2073163 and rs1155794 from haploblock 2 were associated with 2-hour plasma glucose levels in men during the 3-year follow-up. The same two markers together with rs2073162 associated with the conversion of IGT to T2D in men. The risk of developing T2D was approximately 2-fold in individuals with genotypes associated with higher 2-hour plasma glucose levels; the hazard ratios were 2.192 (p = 0.025) for rs2073162-A, 2.191 (p = 0.027) for rs2073163-C, and 1.998 (p = 0.054) for rs1155974-T. These results suggest that TNMD polymorphisms are associated with adiposity and also with glucose metabolism and conversion from IGT to T2D in men.
    # Example Output: ```
    [
        ("glucose", "rs5966709", False),
        ("glucose", "rs4828037", False),
        ("glucose", "rs11798018", False),
        ("glucose", "rs2073163", True),
        ("glucose", "rs1155794", True),
        ("glucose", "rs2073162", False),
        ("glucose", "rs1155974", False)
    ]
    ```
"""

# ----------------------------
# Annotation guideline snippets used for {relation_type_guideline} placeholder in step_2
# # One fixed example is presented in prompt for demonstration, while dynamic few-shot examples can also be applied here.
# ----------------------------
disease_chemical_relation_type = """
    Types of Relationships you should identify between DiseaseOrPhenotypicFeature and ChemicalEntity entities:
    - Positive_Correlation: 
        * Chemical-induced disease.
        * Chemical (or Higher dose of the chemical) causes a higher risk of the disease.
        * Disease causes the increase of the chemical measured level.
        * The level of the chemical and the risk of the disease present a positive correlation.
        * Chemical exposures during development alter disease susceptibility later in life.
    - Negative_Correlation:
        * The disease-treated chemical/drug.
        * Disease causes the decrease of the chemical measured level.
        * The chemical/drug drops down the susceptibility of the disease.
    - Association
        * A safety drug of the potential disease (PMID:20722491 - capecitabine(C110904) - hepatic and renal dysfunctions(D008107|D007674))
        * The associations of the pairs which cannot be categorized to positive/negative correlation.
        * The associations without clear description.
   
    
    # Example Input Entities: 
    [
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Creatinine"),
        ("Renal dysfunction (nephrotoxic)", "Cyclosporine"),
        ("Renal dysfunction (nephrotoxic)", "Tacrolimus"),
        ("Hepatorenal syndrome", "Creatinine")
    ]
    # Example Input Text: End-stage renal disease (ESRD) after orthotopic liver transplantation (OLTX) using calcineurin-based immunotherapy: risk of development and treatment. BACKGROUND: The calcineurin inhibitors cyclosporine and tacrolimus are both known to be nephrotoxic. Their use in orthotopic liver transplantation (OLTX) has dramatically improved success rates. Recently, however, we have had an increase of patients who are presenting after OLTX with end-stage renal disease (ESRD). This retrospective study examines the incidence and treatment of ESRD and chronic renal failure (CRF) in OLTX patients. METHODS: Patients receiving an OLTX only from June 1985 through December of 1994 who survived 6 months postoperatively were studied (n=834). Our prospectively collected database was the source of information. Patients were divided into three groups: Controls, no CRF or ESRD, n=748; CRF, sustained serum creatinine >2.5 mg/dl, n=41; and ESRD, n=45. Groups were compared for preoperative laboratory variables, diagnosis, postoperative variables, survival, type of ESRD therapy, and survival from onset of ESRD. RESULTS: At 13 years after OLTX, the incidence of severe renal dysfunction was 18.1% (CRF 8.6% and ESRD 9.5%). Compared with control patients, CRF and ESRD patients had higher preoperative serum creatinine levels, a greater percentage of patients with hepatorenal syndrome, higher percentage requirement for dialysis in the first 3 months postoperatively, and a higher 1-year serum creatinine. Multivariate stepwise logistic regression analysis using preoperative and postoperative variables identified that an increase of serum creatinine compared with average at 1 year, 3 months, and 4 weeks postoperatively were independent risk factors for the development of CRF or ESRD with odds ratios of 2.6, 2.2, and 1.6, respectively. Overall survival from the time of OLTX was not significantly different among groups, but by year 13, the survival of the patients who had ESRD was only 28.2% compared with 54.6% in the control group. Patients developing ESRD had a 6-year survival after onset of ESRD of 27% for the patients receiving hemodialysis versus 71.4% for the patients developing ESRD who subsequently received kidney transplants. CONCLUSIONS: Patients who are more than 10 years post-OLTX have CRF and ESRD at a high rate. The development of ESRD decreases survival, particularly in those patients treated with dialysis only. Patients who develop ESRD have a higher preoperative and 1-year serum creatinine and are more likely to have hepatorenal syndrome. However, an increase of serum creatinine at various times postoperatively is more predictive of the development of CRF or ESRD. New strategies for long-term immunosuppression may be needed to decrease this complication.
    # Example Output: 
    [
        ("End-stage renal disease (ESRD, chronic renal failure, CRF)", "Creatinine", "Positive_Correlation"),
        ("Renal dysfunction (nephrotoxic)", "Cyclosporine", "Positive_Correlation"),
        ("Renal dysfunction (nephrotoxic)", "Tacrolimus", "Positive_Correlation"),
        ("Hepatorenal syndrome", "Creatinine", "Positive_Correlation")
    ]

    """

disease_gene_type = """Types of Relationships you should identify between DiseaseOrPhenotypicFeature and GeneOrGeneProduct entities:
    - Positive_Correlation
        * The overdose of protein causes the disease.
        * The knockout gene prevents the disease.
        * The side effect of protein (drug) causes the disease
    - Negative_Correlation
        * The protein (drug) is used to treat/prevent the disease.
        * Lack of the protein causes the disease.
        * The knockout gene causes the disease.
    - Association
        * The associations of the pairs which cannot be categorized to previous relation types.
        * The associations without clear description.
        * The functional gene prevents the occurrence of the disease.
        * Protein deficiency.

    Notice that if an association between a variant and a disease is confirmed, the corresponding gene should associate
    with the disease by “Association” type.
    
    
    # Example Input Entities: ```
    [
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "HNF-6 (Hepatocyte nuclear factor-6)")
    ]
    ```
    # Example Input Text: Hepatocyte nuclear factor-6: associations between genetic variability and type II diabetes and between genetic variability and estimates of insulin secretion. The transcription factor hepatocyte nuclear factor (HNF)-6 is an upstream regulator of several genes involved in the pathogenesis of maturity-onset diabetes of the young. We therefore tested the hypothesis that variability in the HNF-6 gene is associated with subsets of Type II (non-insulin-dependent) diabetes mellitus and estimates of insulin secretion in glucose tolerant subjects.   We cloned the coding region as well as the intron-exon boundaries of the HNF-6 gene. We then examined them on genomic DNA in six MODY probands without mutations in the MODY1, MODY3 and MODY4 genes and in 54 patients with late-onset Type II diabetes by combined single strand conformational polymorphism-heteroduplex analysis followed by direct sequencing of identified variants. An identified missense variant was examined in association studies and genotype-phenotype studies.   We identified two silent and one missense (Pro75 Ala) variant. In an association study the allelic frequency of the Pro75Ala polymorphism was 3.2% (95% confidence interval, 1.9-4.5) in 330 patients with Type II diabetes mellitus compared with 4.2% (2.4-6.0) in 238 age-matched glucose tolerant control subjects. Moreover, in studies of 238 middle-aged glucose tolerant subjects, of 226 glucose tolerant offspring of Type II diabetic patients and of 367 young healthy subjects, the carriers of the polymorphism did not differ from non-carriers in glucose induced serum insulin or C-peptide responses.   Mutations in the coding region of the HNF-6 gene are not associated with Type II diabetes or with changes in insulin responses to glucose among the Caucasians examined.
    # Example Output: 
        [
    ("Type II diabetes (Type II diabetes mellitus, Type II diabetic, maturity-onset diabetes)", "HNF-6 (Hepatocyte nuclear factor-6)", "Association"),
    ]

    """


disease_variant_type ="""
Types of Relationships you should identify between DiseaseOrPhenotypicFeature and SequenceVariant entities:
    - Positive_Correlation
        * The variant increases the risk of the disease.
        * Significant frequency (p-value) of the disease with the specific allele.
        * The variant causes the gene to be either over-express or non-functional and further causes disease (or raises the disease susceptibility).
        * Disease is caused by protein deficiency, and the variant is responsible for the deficiency.
        * The variant plays a role in genetic predisposition to the disease.
        * The variant has a significant contribution to the disease.
        * Annotate “Positive_Correlation” to the pair of “founder mutation” and the target disease.(PMID:10788334,19394258)
    - Negative_Correlation
        * The variant decreases the risk of the disease.
    - Association
        * The variant observed from a number of patients.
        * The variant is associated with a lower prevalence of the disease or is responsible for the lower disease susceptibility
        * The associations of the pairs which cannot be categorized to “Cause” relation type.
        * The associations without clear description.

    # Example Input Entities: 
    [
        ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "5382insC"),
        ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "C61G (rs28897672)"),
        ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "4153delA"),
        ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "5382insC"),
        ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "C61G (rs28897672)"),
        ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "4153delA"),
        ("ovarian cancer (ovarian cancers)", "5382insC"),
        ("ovarian cancer (ovarian cancers)", "C61G (rs28897672)"),
        ("ovarian cancer (ovarian cancers)", "4153delA"),
        ("BRCA1 abnormalities", "5382insC"),
        ("BRCA1 abnormalities", "C61G (rs28897672)"),
        ("BRCA1 abnormalities", "4153delA")
    ]

    # Example Input Text: End-stage renal disease (ESRD) after orthotopic liver transplantation (OLTX) using calcineurin-based immunotherapy: risk of development and treatment. BACKGROUND: The calcineurin inhibitors cyclosporine and tacrolimus are both known to be nephrotoxic. Their use in orthotopic liver transplantation (OLTX) has dramatically improved success rates. Recently, however, we have had an increase of patients who are presenting after OLTX with end-stage renal disease (ESRD). This retrospective study examines the incidence and treatment of ESRD and chronic renal failure (CRF) in OLTX patients. METHODS: Patients receiving an OLTX only from June 1985 through December of 1994 who survived 6 months postoperatively were studied (n=834). Our prospectively collected database was the source of information. Patients were divided into three groups: Controls, no CRF or ESRD, n=748; CRF, sustained serum creatinine >2.5 mg/dl, n=41; and ESRD, n=45. Groups were compared for preoperative laboratory variables, diagnosis, postoperative variables, survival, type of ESRD therapy, and survival from onset of ESRD. RESULTS: At 13 years after OLTX, the incidence of severe renal dysfunction was 18.1% (CRF 8.6% and ESRD 9.5%). Compared with control patients, CRF and ESRD patients had higher preoperative serum creatinine levels, a greater percentage of patients with hepatorenal syndrome, higher percentage requirement for dialysis in the first 3 months postoperatively, and a higher 1-year serum creatinine. Multivariate stepwise logistic regression analysis using preoperative and postoperative variables identified that an increase of serum creatinine compared with average at 1 year, 3 months, and 4 weeks postoperatively were independent risk factors for the development of CRF or ESRD with odds ratios of 2.6, 2.2, and 1.6, respectively. Overall survival from the time of OLTX was not significantly different among groups, but by year 13, the survival of the patients who had ESRD was only 28.2% compared with 54.6% in the control group. Patients developing ESRD had a 6-year survival after onset of ESRD of 27% for the patients receiving hemodialysis versus 71.4% for the patients developing ESRD who subsequently received kidney transplants. CONCLUSIONS: Patients who are more than 10 years post-OLTX have CRF and ESRD at a high rate. The development of ESRD decreases survival, particularly in those patients treated with dialysis only. Patients who develop ESRD have a higher preoperative and 1-year serum creatinine and are more likely to have hepatorenal syndrome. However, an increase of serum creatinine at various times postoperatively is more predictive of the development of CRF or ESRD. New strategies for long-term immunosuppression may be needed to decrease this complication.
    # Example Output: 
    [
        ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "5382insC", "Positive_Correlation"),
        ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "C61G (rs28897672)", "Positive_Correlation"),
        ("breast-ovarian cancer (breast-ovarian cancer syndrome)", "4153delA", "Positive_Correlation"),
        ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "5382insC", "Positive_Correlation"),
        ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "C61G (rs28897672)", "Positive_Correlation"),
        ("breast cancer (breast or ovarian cancer; breast and ovarian cancers; breast and ovarian cancer)", "4153delA", "Positive_Correlation"),
        ("ovarian cancer (ovarian cancers)", "5382insC", "Positive_Correlation"),
        ("ovarian cancer (ovarian cancers)", "C61G (rs28897672)", "Positive_Correlation"),
        ("ovarian cancer (ovarian cancers)", "4153delA", "Positive_Correlation"),
        ("BRCA1 abnormalities", "5382insC", "Positive_Correlation"),
        ("BRCA1 abnormalities", "C61G (rs28897672)", "Positive_Correlation"),
        ("BRCA1 abnormalities", "4153delA", "Positive_Correlation")
    ]

"""

gene_gene_type = """
    Types of Relationships you should identify between two GeneOrGeneProduct entities:
    ★ Positive_Correlation
        ❏ Two genes present the positive correlation in gene expression results.
        ❏ Gene A is a transcription factor of the gene B, and gene A upper regulates the gene B.
        ❏ Two genes present a positive correlation in any way.
    ★ Negative_Correlation
        ❏ Two genes present the negative correlation in gene expression results.
        ❏ Gene A is a transcription factor of the gene B, and gene A down regulates the gene B.
        ❏ Two genes present a negative correlation in any way.
    ★ Bind
        ❏ physical interaction between two proteins.
        ❏ Protein A binds the promoter of gene B.
        ❏ Two or more proteins in a complex.
        ❏ Annotate “bind” to the protein and its protein receptor. ("androgen receptor" and "androgen" in
    PMID:15599941)
        ❏ If the binding causes a positive or negative correlation, annotate the association to Positive_Correlation/Negative_Correlation.
        ❏ DO NOT annotate the protein bind on a gene promoter region.
    ★ Association
        ❏ Annotate the modification (e.g, phosphorylation, dephosphorylation, acetylation ,deacetylation and other modifications) to association.
        ❏ The associations of the pairs which cannot be categorized to any other association types.
        ❏ The associations without clear description.

    Example Input Entities: ```
    [
        ("IL1B (interleukin 1 beta, IL1beta, Proinflammatory cytokine)", "NFkappaB (nuclear factor of kappa B)"),
    ]
    ```
    Example Input Text: A polymorphism of C-to-T substitution at -31 IL1B is associated with the risk of advanced gastric adenocarcinoma in a Japanese population. Proinflammatory cytokine gene polymorphisms have been demonstrated to associate with gastric cancer risk, of which IL1B-31T/C and -511C/T changes have been well investigated due to the possibility that they may alter the IL1B transcription. The signal transduction target upon interleukin 1 beta (IL1beta) stimulation, the nuclear factor of kappa B (NFkappaB) activation, supports cancer development, signal transduction in which is mediated by FS-7 cell-associated cell surface antigen (FAS) signaling. Based on recent papers describing the prognostic roles of the polymorphisms and the NFkappaB functions on cancer development, we sought to determine if Japanese gastric cancer patients were affected by the IL1B -31/-511 and FAS-670 polymorphisms. A case-control study was conducted on incident gastric adenocarcinoma patients (n=271) and age-gender frequency-matched control subjects (n=271). We observed strong linkage disequilibrium between the T allele at -511 and the C allele at -31 and between the C allele at -511 and the T allele at -31 in IL1B in both the cases and controls (R (2)=0.94). Neither IL1B-31, -511 nor FAS-670 polymorphisms showed significantly different risks of gastric adenocarcinoma. Though FAS-670 polymorphisms did not show any significant difference, the proportion of subjects with IL1B-31TT (or IL1B-511CC) increased according to stage (trend P=0.019). In particular, subjects with stage IV had a two times higher probability of having either IL1B-31TT (or IL1B-511CC) genotype compared with stage I subjects. These observations suggest that IL1B-31TT and IL1B-511CC are associated with disease progression.
    Example Output: ```
    [
        ("IL1B (interleukin 1 beta, IL1beta, Proinflammatory cytokine)", "NFkappaB (nuclear factor of kappa B)", "Association")
    ]
    ```
"""

gene_chemical_type = """
    Types of Relationships you should identify between GeneOrGeneProduct and ChemicalEntity entities:
    (Notice: If an association between a variant and a chemical is confirmed, the same association type should be assigned to the corresponding gene of the variant and the chemical.)
    ★ Positive_Correlation
        ❏ The chemical causes a higher expression of the gene.
        ❏ Higher gene expression causes higher sensitivity of the chemical.
        ❏ The variant triggers the chemical adverse effects or causes the side effect worse.
        ❏ The gene causes the chemical over-response.
        ❏ Two genes present a positive correlation in any way.
    ★ Negative_Correlation
        ❏ The chemical causes a lower expression of the gene.
        ❏ Higher gene expression causes lower sensitivity (resistance) of the chemical, and vise versa.
        ❏ The variant has a protective role for the development of the chemical adverse effects.
        ❏ The gene causes the chemical resistance.
        ❏ Two genes present a negative correlation in any way.
    ★ Association
        ❏ The associations of the pairs which cannot be categorized to any other association types.
        ❏ The associations without clear description.
    ★ Bind
        ❏ A chemical binds the promoter of a gene.
        ❏ A protein is the chemical receptor.
        ❏ If the binding causes a positive or negative correlation, annotate the association to Positive_Correlation/Negative_Correlation.

    # Example Input Entities: ```
    [
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "Chloroacetaldehyde"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "cisplatin")
    ]
    ```
    # Example Input Text: Chloroacetaldehyde as a sulfhydryl reagent: the role of critical thiol groups in ifosfamide nephropathy. Chloroacetaldehyde (CAA) is a metabolite of the alkylating agent ifosfamide (IFO) and putatively responsible for renal damage following anti-tumor therapy with IFO. Depletion of sulfhydryl (SH) groups has been reported from cell culture, animal and clinical studies. In this work the effect of CAA on human proximal tubule cells in primary culture (hRPTEC) was investigated. Toxicity of CAA was determined by protein content, cell number, LDH release, trypan blue exclusion assay and caspase-3 activity. Free thiols were measured by the method of Ellman. CAA reduced hRPTEC cell number and protein, induced a loss in free intracellular thiols and an increase in necrosis markers. CAA but not acrolein inhibited the cysteine proteases caspase-3, caspase-8 and cathepsin B. Caspase activation by cisplatin was inhibited by CAA. In cells stained with fluorescent dyes targeting lysosomes, CAA induced an increase in lysosomal size and lysosomal leakage. The effects of CAA on cysteine protease activities and thiols could be reproduced in cell lysate. Acidification, which slowed the reaction of CAA with thiol donors, could also attenuate effects of CAA on necrosis markers, thiol depletion and cysteine protease inhibition in living cells. Thus, CAA directly reacts with cellular protein and non-protein thiols, mediating its toxicity on hRPTEC. This effect can be reduced by acidification. Therefore, urinary acidification could be an option to prevent IFO nephropathy in patients.
    # Example Output: ```
    [
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "Chloroacetaldehyde", "Negative_Correlation"),
        ("cysteine proteases (caspase-3, caspase-8, cathepsin B, Caspase, cysteine protease)", "cisplatin", "Positive_Correlation")
    ]

    ```
"""

chemical_chemical_type = """
    Types of Relationships you should identify between two ChemicalEntity entities:
    ★ Positive_Correlation (between A and B)
        ❏ Chemical A increases the sensitivity of the chemical B.
        ❏ Chemical A increases the treatment/inducing effectiveness of the chemical B to a disease.
        ❏ Chemical A increases the effectiveness of the gene activation caused by chemical B.
    ★ Negative_Correlation (between A and B)
        ❏ Chemical A decreases the sensitivity of the chemical B.
        ❏ Chemical A decreases the treatment/inducing effectiveness of the chemical B to a disease.(PMID:25080425 betaine attenuates isoproterenol-induced acute myocardial injury in rats.)
        ❏ Chemical A decreases the side effects of the chemical B. (PMID:24587916 LF(D007781) - Dexamethasone(D003907))
         Chemical A decreases the gene activation caused by chemical B. (PMID:17035713 Caspase activation by cisplatin was inhibited by CAA)
    ★ Association
        ❏ Annotate the chemical conversion to association type. (PMID:17391797 that catalyzes the conversion of phosphatidylethanolamine to phosphatidylcholine.)
        ❏ The associations of the pairs which cannot be categorized to any other association types.
        ❏ The associations without clear description.
    ★ Drug_Interaction
        ❏ A pharmacodynamic interaction occurs when two chemicals/drugs are given together.
    ★ Cotreatment
        ❏ Combination therapy.
    ★ Conversion
        ❏ A chemical converse to the other chemical.

    # Example Input Entities: ```
    [
        ("phosphatidylethanolamine", "phosphatidylcholine")
    ]
    ```
    # Example Input Text: The phosphatidylethanolamine N-methyltransferase gene V175M single nucleotide polymorphism confers the susceptibility to NASH in Japanese population. BACKGROUND/AIMS: The genetic predisposition on the development of nonalcoholic steatohepatitis (NASH) has been poorly understood. A functional polymorphism Val175Met was reported in phosphatidylethanolamine N-methyltransferase (PEMT) that catalyzes the conversion of phosphatidylethanolamine to phosphatidylcholine. The aim of this study was to investigate whether the carriers of Val175Met variant impaired in PEMT activity are more susceptible to NASH. METHODS: Blood samples of 107 patients with biopsy-proven NASH and of 150 healthy volunteers were analyzed by the polymerase chain reaction (PCR) and restriction fragment length polymorphism. RESULTS: Val175Met variant allele of the PEMT gene was significantly more frequent in NASH patients than in healthy volunteers (p<0.001), and carriers of Val175Met variant were significantly more frequent in NASH patients than in healthy volunteers (p<0.01). Among NASH patients, body mass index was significantly lower (p<0.05), and non-obese patients were significantly more frequent (p<0.001) in carriers of Val175Met variant than in homozygotes of wild type PEMT. CONCLUSIONS: Val175Met variant of PEMT could be a candidate molecule that determines the susceptibility to NASH, because it is more frequently observed in NASH patients and non-obese persons with Val175Met variant of PEMT are facilitated to develop NASH.
    # Example Output: ```
    [
        ("phosphatidylethanolamine", "phosphatidylcholine", Conversion)
    ]
    ```
"""

chemical_variant_type = """
    Types of Relationships you should identify between ChemicalEntity and SequenceVariant entities:
    ★ Positive_Correlation
        ❏ The chemical causes a higher expression of the gene because of the specific variant.
        ❏ The variant causes higher sensitivity of the chemical.
        ❏ The variant causes the chemical over-response.
        ❏ The gene and the chemical present a positive correlation in any way.
    ★ Negative_Correlation
        ❏ The chemical causes a lower expression of the gene because of the specific variant.
        ❏ The variant causes lower sensitivity of the chemical.
        ❏ The variant causes the chemical resistance.
        ❏ The gene and the chemical present a negative correlation in any way.
        ❏ The variant presents a protective role to the chemical adverse effect (or chemical caused disease).
    ★ Association
        ❏ The associations of the pairs which cannot be categorized to positive/negative correlation.
        ❏ The associations without clear description.
        ❏ variant located on a chemical specific binding site, e.g., the sequence variant c.465G>T encodes a conservative amino acid substitution, p.Glu155Asp, located in EF-hand 4, the calcium binding site of GCAP2 protein. (PMID:21405999)

    # Example Input Entities: ```
    [
        ("glucose", "rs2073163"),
        ("glucose", "rs1155794"),
    ]
    ```
    Example Input Text: Tenomodulin is associated with obesity and diabetes risk: the Finnish diabetes prevention study. We recently showed that long-term weight reduction changes the gene expression profile of adipose tissue in overweight individuals with impaired glucose tolerance (IGT). One of the responding genes was X-chromosomal tenomodulin (TNMD), a putative angiogenesis inhibitor. Our aim was to study the associations of individual single nucleotide polymorphisms and haplotypes with adiposity, glucose metabolism, and the risk of type 2 diabetes (T2D). Seven single nucleotide polymorphisms from two different haploblocks were genotyped from 507 participants of the Finnish Diabetes Prevention Study (DPS). Sex-specific genotype effects were observed. Three markers of haploblock 1 were associated with features of adiposity in women (rs5966709, rs4828037) and men (rs11798018). Markers rs2073163 and rs1155794 from haploblock 2 were associated with 2-hour plasma glucose levels in men during the 3-year follow-up. The same two markers together with rs2073162 associated with the conversion of IGT to T2D in men. The risk of developing T2D was approximately 2-fold in individuals with genotypes associated with higher 2-hour plasma glucose levels; the hazard ratios were 2.192 (p = 0.025) for rs2073162-A, 2.191 (p = 0.027) for rs2073163-C, and 1.998 (p = 0.054) for rs1155974-T. These results suggest that TNMD polymorphisms are associated with adiposity and also with glucose metabolism and conversion from IGT to T2D in men.
    Example Output: ```
    [
        ("glucose", "rs2073163", "Association"),
        ("glucose", "rs1155794", "Association")
    ]
    ```
"""

# ----------------------------
# prompt snippets used for step_3 (rule validation)
# ----------------------------
others_exception = """
\n# OTHER EXCEPTIONS
❏ Co-occurrence in the same sentence is not required.
❏ DO NOT annotate a confirmed irrelevance. (e.g., “Nonetheless, NBS1 gene heterozygosity is not a major risk factor for lymphoid malignancies in childhood and adolescence” in PMID:16152606)
❏ Annotate the associations with explicit statements in the text.
❏ The full text is NOT allowed to access. DO NOT annotate if the statement in the abstract is not clear. (e.g.,PMID:14722929)
❏ For Co-treatment, we annotate “Negative_Correlation” between the two chemicals and the disease, and a “Cotreatment” relation between the two chemicals. For an example in PMID:16720068, “Possible neuroleptic malignant syndrome related to concomitant treatment with paroxetine and alprazolam.”.
❏ For Drug_Interaction, we annotate “Positive_Correlation” between the two chemicals and the disease, and a “Drug_Interaction” relation between the two chemicals.
❏ DO NOT annotate the pairs of concepts with negative conclusions (e.g., “not significant”, “no correlation was observed”). For instance, “For most of the variants identified in the Kenyan and Sudanese study population, a causative association with NSARD appears to be unlikely” is not annotated.
❏ In the case that a disease is induced by a chemical, and the other concepts (e.g., chemical, or protein) treats/affects the induced disease, we annotate the three pairs by following examples. For instance:
    ○ “In addition, there is convincing clinical evidence that monotherapy with continuous subcutaneous apomorphine infusions is associated with marked reductions of preexisting levodopa-induced dyskinesias.” (PMID:11009181)
❏ Positive_Correlation between levodopa and dyskinesias
❏ Negative_Correlation between apomorphine and dyskinesias
❏ Negative_Correlation between apomorphine and levodopa
    ○ Absence of PKC-alpha attenuates lithium-induced nephrogenic diabetes insipidus. (PMID:25006961)
❏ Positive_Correlation between lithiumand nephrogenic diabetes
❏ Positive_Correlation between PKC-alpha and dyskinesias
❏ Positive_Correlation between PKC-alpha and lithium
    ○ Characterization of a novel BCHE "silent" allele: point mutation (p.Val204Asp) causes loss of activity and prolonged apnea with suxamethonium. (PMID:25054547)
❏ Positive_Correlation between p.Val204Asp and apnea
❏ Negative_Correlation between apnea and suxamethonium
❏ Association between p.Val204Asp and suxamethonium
❏ Association between BCHE and apnea
❏ Association between BCHE and suxamethonium
    ○ Curcumin prevents maleate-induced nephrotoxicity (PMID:25119790)
❏ Negative_Correlation between Curcumin and nephrotoxicity
❏ Positive_Correlation between maleate and nephrotoxicity
❏ Negative_Correlation between Curcumin and maleate
❏ Annotate the previous reported association, even the conclusion demonstrates the association is not significant.

"""

disease_chemical_validation = """
# Relation Pairs
❏ Annotate the association of chemicals and disease due to the “toxicity” and “drug-induced”, but not annotate the chemicals with the disease concept-“toxicity” directly.
    ○ PMID:19728177: “RESULTS: Our patient was admitted to the MICU after being found unresponsive with presumed toxicity from acetaminophen which was ingested over a 2-day period. ... In patients with FHF and cerebral edema from acetaminophen overdose, prolonged therapeutic hypothermia could potentially be used as a life saving therapy and a bridge to hepatic and neurological recovery.” We annotate the associations between acetaminophen and FHF/cerebral edema due to the toxicity, but we do not annotate acetaminophen and toxicity.
❏ Annotate the relation between the target disease and the measure level of the molecules/chemicals in blood test with correlation, “Compared with control patients, CRF and ESRD patients had higher preoperative serum creatinine levels, a greater percentage of patients with hepatorenal syndrome.”, the pairs of “CRF, creatinine” and “ESRD, creatinine” is annotated. (PMID:11773892)
❏ DO NOT annotate the association of disease to the staining chemicals. For instance, “The extent of neuronal injury was determined by 2,3,5-triphenyltetrazolium staining.” (PMID:1711760) and “Renal lesions were analyzed in hematoxylin and eosin, periodic acid-Schiff, and Masson's trichrome stains. SRL-treated rats presented proteinuria and NGAL (serum and urinary) as the best” (PMID:24971338)
❏ DO NOT annotate the association of any disease with the chemical concept “analgesia” in “patient-controlled analgesia (PCA) pump”. For instance, “patient-controlled analgesia (PCA) pump” (PMID:9672936)
❏ DO NOT annotate toxicity with its chemical. For instance, do not annotate “unresponsive with presumed toxicity from acetaminophen” (PMID:19728177)
❏ Annotate the chemicals which prevents the chemical-induced disease (PMID:24971338+18503483)

# Disease-chemical Relation Types:
★ Positive_Correlation
    ❏ Chemical-induced disease.
    ❏ Chemical (or Higher dose of the chemical) causes a higher risk of the disease.
    ❏ Disease causes the increase of the chemical measured level.
    ❏ The level of the chemical and the risk of the disease present a positive correlation.
    ❏ Chemical exposures during development alter disease susceptibility later in life.
★ Negative_Correlation
    ❏ The disease-treated chemical/drug.
    ❏ Disease causes the decrease of the chemical measured level.
    ❏ The chemical/drug drops down the susceptibility of the disease.
★ Association
    ❏ A safety drug of the potential disease (PMID:20722491 - capecitabine(C110904) - hepatic and renal dysfunctions(D008107|D007674))
    ❏ The associations of the pairs which cannot be categorized to positive/negative correlation.
    ❏ The associations without clear description.
""" + others_exception

disease_gene_validation = """
# Relation Pairs
❏ DO NOT annotate the prognostic factor of genes or its variants to the target disease.
❏ Annotate the negative correlation (e.g., knockdown gene to the target disease). For instance, annotates Gankyrin and HCC in “Gankyrin expression in the tumor microenvironment is negatively correlated with progression-free survival in patients undergoing sorafenib treatment for HCC.” (PMID:28777492)
❏ DO NOT annotate the symptoms of the target disease to the correlated variants or genes, unless the clear statement
between the symptom and variant is provided.
# Disease-Gene Relation Types:
❏ If an association between a variant and a disease is confirmed, the corresponding gene should associate with the disease by “Association” type.
★ Positive_Correlation
    ❏ The overdose of protein causes the disease.
    ❏ The knockout gene prevents the disease.
    ❏ The side effect of protein (drug) causes the disease
★ Negative_Correlation
    ❏ The protein (drug) is used to treat/prevent the disease.
    ❏ Lack of the protein causes the disease.
    ❏ The knockout gene causes the disease.
★ Association
    ❏ The associations of the pairs which cannot be categorized to previous relation types.
    ❏ The associations without clear description.
    ❏ The functional gene prevents the occurrence of the disease.
    ❏ Protein deficiency.
"""+ others_exception

disease_variant_validation = """
# Relation Pairs
❏ Annotate the corresponding diseases with the variant when it is observed in patients.
❏ Annotate the diseases with the associated genes.
❏ Annotate the corresponding gene of the variant with the variant-associated disease.
❏ Annotate the association between the variants (and corresponding gene) and the diseases belonging to the target disease. For instance, annotate the association between “autosomal recessive disease” and “272gly----stop” (PMID:1671881).
❏ Annotate the minor association (e.g., the observation of the association on few patients).
❏ Annotate the associations in the particular races.
❏ Annotate the association between disease and variant if the inheritance of the genetic disease is confirmed by the variant.
❏ DO NOT annotate the symptoms of the target disease to the correlated variants or genes, unless the clear statement between the symptom and variant is provided. (ex in PMID: 1952108)

# Disease-Variant Relation Types:
★ Positive_Correlation
    ❏ The variant increases the risk of the disease.
    ❏ Significant frequency (p-value) of the disease with the specific allele.
    ❏ The variant causes the gene to be either over-express or non-functional and further causes disease (orraises the disease susceptibility).
    ❏ Disease is caused by protein deficiency, and the variant is responsible for the deficiency.
    ❏ The variant plays a role in genetic predisposition to the disease.
    ❏ The variant has a significant contribution to the disease.
    ❏ Annotate “Positive_Correlation” to the pair of “founder mutation” and the target disease. (PMID:10788334,19394258)
★ Negative_Correlation
    ❏ The variant decreases the risk of the disease.
★ Association
    ❏ The variant observed from a number of patients.
    ❏ The variant is associated with a lower prevalence of the disease or is responsible for the lower disease susceptibility
    ❏ The associations of the pairs which cannot be categorized to “Cause” relation type.
    ❏ The associations without clear description.
"""

gene_gene_validation = """
# Relation Pairs
❏ Annotate the pairs of the members in the protein complex, e.g., EPO/EPOR (PMID:22808010).
    ○ In “We cloned the cDNA of three remaining human NADH:ubiquinone oxidoreductase subunits of this IP fraction: the NDUFS2 (49 kDa), NDUFS3 (30 kDa), and NDUFS6 (13 kDa) subunits.”, all the pairs between two of the NDUFS2, NDUFS3, and NDUFS6 are annotated. (PMID:9647766)
❏ DO NOT annotate the “is_a” association between two genes. For instance, DO NOT annotate ”breast and ovarian cancer gene” and “BRCA1” (PMID:8944024).
❏ Annotate protein-protein interaction for singling proteins and the receptors. For instance, “Epo-R signaling proteins (Akt, STAT5, p70s6k, LYN, and p38MAPK)” (PMID:27640183)

# Gene-Gene Relation Types:
★ Positive_Correlation
    ❏ Two genes present the positive correlation in gene expression results.
    ❏ Gene A is a transcription factor of the gene B, and gene A upper regulates the gene B.
    ❏ Two genes present a positive correlation in any way.
★ Negative_Correlation
    ❏ Two genes present the negative correlation in gene expression results.
    ❏ Gene A is a transcription factor of the gene B, and gene A down regulates the gene B.
    ❏ Two genes present a negative correlation in any way.
★ Bind
    ❏ physical interaction between two proteins.
    ❏ Protein A binds the promoter of gene B.
    ❏ Two or more proteins in a complex.
    ❏ Annotate “bind” to the protein and its protein receptor. ("androgen receptor" and "androgen" in PMID:15599941)
    ❏ If the binding causes a positive or negative correlation, annotate the association to Positive_Correlation/Negative_Correlation.
    ❏ DO NOT annotate the protein bind on a gene promoter region.
★ Association
    ❏ Annotate the modification (e.g, phosphorylation, dephosphorylation, acetylation ,deacetylation and other modifications) to association.
    ❏ The associations of the pairs which cannot be categorized to any other association types.
    ❏ The associations without clear description.
"""

gene_chemical_validation = """
# Relation Pairs
❏ Annotate the chemical with the gene receptor. For instance, "antipsychotic drugs that have a high affinity with the
D2 receptor” (PMID:16867246)

# Gene-Chemical Relation Types:
    ❏ If an association between a variant and a chemical is confirmed, the same association type should be assigned to the corresponding gene of the variant and the chemical.
★ Positive_Correlation
    ❏ The chemical causes a higher expression of the gene.
    ❏ Higher gene expression causes higher sensitivity of the chemical.
    ❏ The variant triggers the chemical adverse effects or causes the side effect worse.
    ❏ The gene causes the chemical over-response.
    ❏ Two genes present a positive correlation in any way.
★ Negative_Correlation
    ❏ The chemical causes a lower expression of the gene.
    ❏ Higher gene expression causes lower sensitivity (resistance) of the chemical, and vise versa.
    ❏ The variant has a protective role for the development of the chemical adverse effects.
    ❏ The gene causes the chemical resistance.
    ❏ Two genes present a negative correlation in any way.
★ Association
    ❏ The associations of the pairs which cannot be categorized to any other association types.
    ❏ The associations without clear description.
★ Bind
    ❏ A chemical binds the promoter of a gene.
    ❏ A protein is the chemical receptor.
    ❏ If the binding causes a positive or negative correlation, annotate the association to Positive_Correlation/Negative_Correlation.
"""

chemical_chemical_validation = """
# Relation Pairs
❏ DO NOT annotate the “is_a” association between two chemicals. For instance, “Nefiracetam is a novel pyrrolidone derivative” (PMID:8829135)
❏ Annotate the chemical-chemical association for the chemical contributes to the other for treatment, side effects and drug-drug interaction. In “the midline serotonin B3 cells in the medulla contribute to the hypotensive action of methyldopa”, the pair of serotonin and methyldopa is annotated. (PMID:2422478) 

# Chemical-Chemical Relation Types:
★ Positive_Correlation (between A and B)
    ❏ Chemical A increases the sensitivity of the chemical B.
    ❏ Chemical A increases the treatment/inducing effectiveness of the chemical B to a disease.
    ❏ Chemical A increases the effectiveness of the gene activation caused by chemical B.
★ Negative_Correlation (between A and B)
    ❏ Chemical A decreases the sensitivity of the chemical B.
    ❏ Chemical A decreases the treatment/inducing effectiveness of the chemical B to a disease.(PMID:25080425 betaine attenuates isoproterenol-induced acute myocardial injury in rats.)
    ❏ Chemical A decreases the side effects of the chemical B. (PMID:24587916 LF(D007781) - Dexamethasone(D003907))
    ❏ Chemical A decreases the gene activation caused by chemical B. (PMID:17035713 Caspase activation by cisplatin was inhibited by CAA)
★ Association
    ❏ Annotate the chemical conversion to association type. (PMID:17391797 that catalyzes the conversion of phosphatidylethanolamine to phosphatidylcholine.)
    ❏ The associations of the pairs which cannot be categorized to any other association types.
    ❏ The associations without clear description.
★ Drug_Interaction
    ❏ A pharmacodynamic interaction occurs when two chemicals/drugs are given together.
★ Cotreatment
    ❏ Combination therapy.
★ Conversion
    ❏ A chemical converse to the other chemical.
"""

chemical_variant_validation = """
# Relation Pairs
❏ Annotate the chemical with the variant, if the chemical binding ability of the gene impaired by the variant. For instance, “This variant (apoE Guangzhou) may cause a marked molecular conformational change of the apoE and thus impair its binding ability to lipids.” (PMID:18046082)
❏ Annotate the chemical with the variant, if different mRNA stability between different alleles affected by the chemical. For instance, “After transfection and inhibition of transcription with actinomycin D, analysis of mRNA turnover failed to reveal differences in mRNA stability between A118 and G118 alleles, indicating a defect in transcription or mRNA maturation." (PMID:16046395)

# Chemical-Variant Relation Types:
★ Positive_Correlation
    ❏ The chemical causes a higher expression of the gene because of the specific variant.
    ❏ The variant causes higher sensitivity of the chemical.
    ❏ The variant causes the chemical over-response.
    ❏ The gene and the chemical present a positive correlation in any way.
★ Negative_Correlation
    ❏ The chemical causes a lower expression of the gene because of the specific variant.
    ❏ The variant causes lower sensitivity of the chemical.
    ❏ The variant causes the chemical resistance.
    ❏ The gene and the chemical present a negative correlation in any way.
    ❏ The variant presents a protective role to the chemical adverse effect (or chemical caused disease).
★ Association
    ❏ The associations of the pairs which cannot be categorized to positive/negative correlation.
    ❏ The associations without clear description.
    ❏ variant located on a chemical specific binding site, e.g., the sequence variant c.465G>T encodes a conservative amino acid substitution, p.Glu155Asp, located in EF-hand 4, the calcium binding site of GCAP2 protein. (PMID:21405999)
"""