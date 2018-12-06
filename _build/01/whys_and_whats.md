---
redirect_from:
  - "/01/whys-and-whats"
title: 'Whys and Whats'
prev_page:
  url: /acknowledgements
  title: 'Acknowledgements'
next_page:
  url: /01/1/example_fitting_time_series
  title: 'Example Fitting Time Series'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /notebooks***"
---
What Is This Book and Why Does It Exist?
====================
*by James Paul Mason*

This is a book that lives at the crossroads of heliophysics and machine learning. This is succinctly captured in the book's title. We authors are all heliophysicists that deal with large quantities of data in our daily work and have stumbled upon the exceptionally applicable tools and techniques of machine learning. This books sets out to show by example some of those real scenarios. 

# Why? 
## Why heliophysics?
Simply put, we now live in a time where there are too much solar and solar-influences data for humans to digest. Recent and upcoming observatories generate petabytes of data, for example, from the [Solar Dynamics Observatory](https://ui.adsabs.harvard.edu/#abs/2012SoPh..275....3P/abstract) launched in 2010 and the upcoming [Daniel K. Inouye Solar Telescope](https://en.wikipedia.org/wiki/Daniel_K._Inouye_Solar_Telescope). The datasets we have access to are varied and rich. The [Heliophysics System Observatory (HSO)](https://www.nasa.gov/content/goddard/heliophysics-system-observatory-hso) is a term that was coined specifically to describe this. It consists of dozens of satellites spanning the solar system. While the HSO focuses on spacecraft, there is no shortage of ground-based observations of the sun and the earth's response to it. Together, these data span at least decades and vary wildly in their resolution in time, space, and wavelength and in their observational target. More measurements of the sun and its impacts exist now than at any time in human history. Nearly all of these data are freely available. There's no indication that the firehose will be constricting in the future. However, the human brain evolves on very slow timescales, so there's little hope that we'll be able to glance at billions of observations and identify the connections and patterns contained within. Fortunately, we're a clever species and are building tools in our machines that can do exactly that. 

## Why machine learning?
As with all computing, machine learning is, at its core, an augmentation of our natural capabilities. In particular, machine learning is good at handling large amount of data, including disparate data and high-dimensional data. That is exactly the situation we find ourselves in with heliophysics data, e.g., the HSO. The strengths of machine learning are specifically designed to fill the niche where our brains do not excel. Of course, this isn't putting us out of a job. The main outputs of machine learning tend to be _identification_ and/or _prediction_, but the _understanding_ still can only be found between keyboard and chair. It is up to us to determine if there is any physical meaning in the results. Nevertheless, we can leverage the huge ML effort in computer science to widen discovery space. Analyzing data in its full dimensionality to find patterns without needing to first reduce it to something we can plot on a screen is a major boon. Machine learning also gives us the tools to remove unimportant information and collapse data to as few dimensions as we like while precisely quantifying what we're losing by doing so. Thus, we can pull from the strengths of both our machines and our brains to make better predictions and gain a deeper understanding of nature. 

## Why this book? 
This book has two main purposes. First, to introduce heliophysicists to machine learning with examples. Secondly and equally, to introduce data scientists to some common heliophysics measurement types and to the broader impact that this research has, again with examples. We hope that heliophysicists will discover how easy machine learning tools are to invoke and be inspired to delve more deeply into it. Similarly, we hope to convince data scientists that in heliophysics, we are in possession of extraordinary datasets ripe for discovery.  

# What?
## What is machine learning?
Machine learning is not just modern computational statistics. The two disciplines were born half a century apart in vastly different computational landscapes. Traditional statistical programming came about when computational resources were highly constrained and as a result many of the techniques rely on various forms of simplification. For example, figuring out an appropriate underlying distribution for data is extremely common. Simplifying assumptions of varying validity are made and its not always easy to quantify the impact of those assumptions. Machine learning, on the other hand, was born in an era where computing is cheap. Assumptions are still made, to be sure, but there's much less restriction on _initial_ assumptions. Instead of the human needing to determine what is important up front, we can just leave all of the numerous features of the data intact. This encourages exploration of data before subtle biases can cut out information that may have lead to new insights. Thus, while both use a computer to get the job done, the disciplines are vastly different in their approach and design. 

Just to get some high-level terminology out of the way as early as possible, see the image below. Machine learning is a sub-discipline of artificial intelligence. It generally requires that the user (the human in the chair) be a part of the overall learning feedback loop, e.g., how to quantify what is important and to determine success. Deep learning goes a step further in removing the human from the feedback loop by taking over that process as well. Thus, a computer can teach itself [how to play Go better than the best humans](https://deepmind.com/blog/alphago-zero-learning-scratch/). More detailed terminology will be discussed subsequently in the examples as the concepts arise.

![From https://www.geospatialworld.net/blogs/difference-between-ai%EF%BB%BF-machine-learning-and-deep-learning/](https://geospatialmedia.s3.amazonaws.com/wp-content/uploads/2017/05/AAEAAQAAAAAAAAhPAAAAJDlkMWMwNTA1LTZkZjUtNDA5MS1hYT.jpg)


## What is heliophysics?
Heliophysics is a term that encompasses a lot. In short, it refers to the physics of the sun and how its light and particles interact with everything in the solar system. There's a lot of stuff in the solar system but our focus tends toward interactions with planetary atmospheres and magnetospheres. This is critically important for us on Earth because "space weather" can become violent and impact many of the technologies we rely on daily. GPS can be taken down. The impact of that stretches far beyond getting directions to your friend's house warming party; farming, oil drilling, and airlines all depend on high-precision Guidance and Navigation Satellite System (GNSS) like GPS. Beyond impacting GPS, flights over the poles have to be rerouted or rescheduled because of the increased radiation. Astronauts have to take shelter for the same reason. In fact, all satellites are vulnerable to solar storms, including communications satellites. Power grids can be overloaded, leading to massive blackouts, as [happened in Quebec in 1989](https://en.wikipedia.org/wiki/March_1989_geomagnetic_storm#Quebec_blackout), which can be deadly when heat isn't available in the winter. There are myriad consequences to severe space weather as detailed in a [2008 National Research Council Report](https://www.nap.edu/catalog/12507/severe-space-weather-events-understanding-societal-and-economic-impacts-a). Fortunately, just as with violent terrestrial weather, with accurate forecasts we can take measures to mitigate these impacts. That is how astronauts know to take shelter before they've already been exposed; how airlines can reroute; how we can put spacecraft into safe modes to better weather the storm. We hope your ears perked up at the mention of forecasting, which is just a more probabilistic term for _prediction_ -- what so much of machine learning is specifically designed to do. 

But heliophysics isn't all about making better space weather forecasts. It's science. Many of us are in it for the joy of discovery and for contributing to humanity's understanding of nature. Not only is it important for understanding the history and future of the solar system, but it is a microscope on the only solar system we know is capable of evolving and harboring life. What if the sun were smaller and dimmer? What if it was much more active -- sending off even bigger disruptive eruptions and more often? These are questions that we have begun to explore in the new hot field of extra-solar planets, or exoplanets. Our ever deepening understanding of heliophysics informs our determination of whether these planets around other stars are potentially habitable. And in exchange, we are provided with context  for our planetary relationship with our star and that most tantalizing question: _Are we alone?_






