## Running Locally
**Disclaimer: These are instructions for the nominal case. If they don't work for you, and you've done some Googling to try to resolve the issue, feel free to [post an issue on the GitHub repository](https://github.com/HelioML/HelioML/issues) and we'll do everything we can to help.**

1. Go to the chapter you're interested in on helioml.org and select the Notebook.
2. Download.
    * At the top of the page, hover over the download icon, and click ".ipynb". This file should download to your local downloads directory.
    * Go to the [HelioML GitHub repository](https://github.com/HelioML/HelioML), right click "environment.yml" and select "Save Link As" to download the file into your local downloads directory.
3. If you don't already have the anaconda distribution of python installed, we highly recommend installing it. [Here are the instructions for that](https://docs.anaconda.com/anaconda/install/).
4. Open up your terminal (or Anaconda Prompt if on Windows) and navigate to the folder containing the files you downloaded in step 2 above (this could be your default download directory or wherever you've decided to move the files).
5. Install. At your command line, type `conda env create -f environment.yml` (we're following the instructions for creating a conda environment from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)). When it prompts you if you want to install all of the detected packages, type `y` and hit enter. It'll take about a minute to install everything.
6. Activate. At your command line, type `conda activate textbook`.
7. Run. Finally, at your command line, type `jupyter notebook` or `jupyter lab` (whichever is your preference). This will open up jupyter in your default browser. Just click on the notebook.ipynb you downloaded and you should now be able to run all of the code locally.


## Running in the Cloud
**Disclaimer: Some of these chapters require a fair amount of compute so be conscious of the resource draw on your cloud computing platform of choice.**

1. Fork the [HelioML repository on GitHub](https://github.com/HelioML/HelioML). [Here are instructions for how to do that if you're not familiar](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) (but it's basically just clicking the "Fork" button in the upper right corner of the repository on GitHub).
2. Load the repository into your cloud computing platform of choice. Here are links for how to do that for a few of the options.
    * [Load into Google Collab](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb)
    * [Load into Binder](https://mybinder.readthedocs.io/en/latest/introduction.html)
    * [Load into Amazon Web Services Elastic Compute Cloud](https://docs.aws.amazon.com/codedeploy/latest/userguide/tutorials-github.html)
    * [Load into Microsoft Azure](https://docs.microsoft.com/en-us/azure/notebooks/quickstart-clone-jupyter-notebook)
