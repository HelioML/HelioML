# Contributing

We consider this book a living document and we're planning to add chapters frequently. If you've used machine learning techniques to analyze any subject matter in heliophysics and published this analysis in a refereed journal, please consider contributing your code! You'll be listed as a contributor to the book and first author of your chapter. 

When contributing a chapter, we recommend the following scientific workflow:

1. **Publish your code with yourself as first author.** The code could be published in a digital repository managed by an institutional library or [published in Zenodo directly via Github](https://guides.github.com/activities/citable-code/). The [SAO/NASA Astrophysics Data System](http://adsabs.harvard.edu/) will automatically index software published in Zenodo via the [Asclepias project](https://github.com/asclepias). Either way, generate a DOI for the code. 

2. **Publish your scientific paper with yourself as first author, and cite the published code in the published paper.** This generates a separate DOI for the published paper (accepted papers are welcome too). AAS Journals' [software policy](https://journals.aas.org/policy-statement-on-software/) contains examples of how to cite software in scientific papers.

3. **Issue a PR to HelioML.** This way, HelioML is not showing anything for the first time. Chapter contributors totally own the first publication of the code and the first publication of the paper. To issue a PR to HelioML:

   1. Fork this entire repository.
   2. Modify the environment.yml file with the libraries you need to run your notebook.
   3. Add another enumerated folder within in the content folder (e.g. HelioML/content/07).
   4. Within this enumerated folder, add two documents: 
       * A .md file that acts as an introduction to the chapter, and
       * A folder named /1 that contains the Jupyter notebook named as `notebook.ipynb` and any ancillary files.  
   5. That's it!

Contributors are required to abide by our [Code of Conduct](https://github.com/HelioML/HelioML/blob/master/CODE_OF_CONDUCT.md).