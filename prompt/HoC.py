prompt = """
This is a multi-label classfication task. Given a reseach article relevant to cancer, your task is to decide zero or more relevant Cancer Hallmarks about the paper based on its title and abstract.
There are 10 cancer hallmarks you will need to decide whether the article is related to. The followings are the cancer hallmarks and their definitions.
1. sustaining proliferative signaling:Cancer cells can initiate and maintain continuous cell division by producing their own growth factors or by altering the sensitivity of receptors to growth factors. Related: Proliferation Receptor Cancer, 	‘Growth factor’ Cancer, ‘Cell cycle’ Cancer.
2. evading growth suppressors:Cancer cells can bypass the normal cellular mechanisms that limit cell division and growth, such as the inactivation of tumor suppressor genes and/or insensitivity to antigrowth signals. Related: ‘Cell cycle’ Cancer, ‘Contact inhibition’
3. resisting cell death:Cancer cells develop resistance to apoptosis, the programmed cell death process, which allows them to survive and continue dividing. Related: Apoptosis Cancer, Necrosis Cancer, Autophagy Cancer.
4. enabling replicative immortality:Cancer cells can extend their ability to divide indefinitely by maintaining the length of telomeres, the protective end caps on chromosomes. Related: Senescence Cancer, Immortalization Cancer
5. inducing angiogenesis:Cancer cells stimulate the growth of new blood vessels, providing the necessary nutrients and oxygen to support their rapid growth. Related: 	Angiogenesis Cancer, ‘Angiogenic factor’
6. activating invasion and metastasis:Cancer cells can invade surrounding tissues and migrate to distant sites in the body, forming secondary tumors called metastases. Related: Metastasis Invasion Cancer
7. genomic instability and mutation:Cancer cells exhibit increased genomic instability, leading to a higher mutation rate, which in turn drives the initiation and progression of cancer. Related: Mutation Cancer, ‘DNA repair’ Cancer, Adducts Cancer, ‘Strand breaks’ Cancer, ‘DNA damage’ Cancer
8. tumor promoting inflammation:Chronic inflammation can promote the development and progression of cancer by supplying growth factors, survival signals, and other molecules that facilitate cancer cell proliferation and survival. Related: Inflammation Cancer, ‘Oxidative stress’ Cancer, Inflammation ‘Immune response’ Cancer
9. cellular energetics:Cancer cells rewire their metabolism to support rapid cell division and growth, often relying more on glycolysis even in the presence of oxygen (a phenomenon known as the Warburg effect). Related: Glycolysis Cancer, ‘Warburg effect’ Cancer
10. avoiding immune destruction:Cancer cells can avoid detection and elimination by the immune system through various mechanisms, such as downregulating cell surface markers or producing immunosuppressive signals. Related: ‘Immune system’ Cancer, Immunosuppression Cancer
"""