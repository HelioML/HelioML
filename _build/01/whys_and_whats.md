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
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---
What Is This Book and Why Does It Exist?
====================
*by James Paul Mason*

This is a book that lives at the crossroads of heliophysics and machine learning. We authors are all heliophysicists that deal with large quantities of data in our daily work and have stumbled upon the exceptionally applicable tools and techniques of machine learning. This books sets out to show by example some of those real scenarios. 

# Why? 
## Why heliophysics?
Simply put, we now live in a time where there are too much solar and solar-influences data for humans to digest. Recent and upcoming observatories generate petabytes of data, for example, from the [Solar Dynamics Observatory](https://ui.adsabs.harvard.edu/#abs/2012SoPh..275....3P/abstract) launched in 2010 and the upcoming [Daniel K. Inouye Solar Telescope](https://en.wikipedia.org/wiki/Daniel_K._Inouye_Solar_Telescope). The datasets we have access to are varied and rich. The term [Heliophysics System Observatory (HSO)](https://www.nasa.gov/content/goddard/heliophysics-system-observatory-hso) was coined specifically to describe this. It consists of dozens of satellites spanning the solar system and observing a variety of different heliophysical phenomena. While the HSO focuses on spacecraft, there is no shortage of ground-based observations of the Sun and the Earth's response to it. Together, these data span decades and vary wildly in their resolution in time, space, and wavelength. More measurements of the Sun and its impacts exist now than at any time in human history. Nearly all of these data are freely available. There's no indication that the firehose will constrict in the future. As a result, there's little hope that humans be able to glance at every single one of these observations and identify the connections and patterns contained within. Fortunately, we're a clever species and are building tools that can do exactly that. 

## Why machine learning?
As with all computing, machine learning is, at its core, an augmentation of our natural capabilities. In particular, machine learning is good at handling large amount of data, including disparate data and high-dimensional data. That is exactly the situation we find ourselves in with heliophysics data. Of course, this isn't putting us out of a job. The main outputs of machine learning tend to be _identification_ and/or _prediction_, but the _understanding_ still can only be found between keyboard and chair. It is up to us to determine if there is any physical meaning in the results. Nevertheless, we can leverage machine learning to widen our discovery space. For example, analyzing data in its full dimensionality to find patterns without needing to first reduce it to something we can plot on a screen is a major boon. Thus, we can pull from the strengths of both our machines and our brains to develop more sophisticated analyses and gain a deeper understanding of nature. 

## Why this book? 
This book has two main purposes. First, to introduce heliophysicists to machine learning with examples. Secondly and equally, to introduce data scientists to some common problems in heliophysics and highlight the broader impact of research on the scientific community at large -- again, with examples. We hope that heliophysicists will discover how easy machine learning tools are to invoke and be inspired to delve more deeply into it. Similarly, we hope to show data scientists that heliophysicists collect extraordinary datasets ripe for discovery.  

# What?
## What is machine learning?
Machine learning is not just modern computational statistics. The two disciplines were born half a century apart in vastly different computational landscapes. Traditional statistical programming came about when computational resources were highly constrained and, as a result, many of the techniques rely on various forms of simplification. A common example is figuring out an appropriate underlying distribution to describe some data. Simplifying assumptions of varying validity are made and its not always easy to quantify the impact of those assumptions. 

Machine learning, on the other hand, became popular in an era where computing is cheap. Assumptions are still made, to be sure, but there's much less restriction on _initial_ assumptions. Instead determining what is important up front, we can leave many, if not all, of the numerous features of the data intact. This encourages exploration of data before subtle biases can cut out information that may have lead to new insights. Thus, while both use a computer to get the job done, the disciplines are vastly different in their approach and design. 

Just to get some high-level terminology out of the way as early as possible, see the image below. Machine learning is a sub-discipline of artificial intelligence. It generally requires that the user (the human in the chair) be a part of the overall learning feedback loop, e.g., how to quantify what is important and to determine success. Deep learning goes a step further in removing the human from the feedback loop by taking over that process as well. Thus, a computer can teach itself [how to play Go better than the best humans](https://deepmind.com/blog/alphago-zero-learning-scratch/). More detailed terminology will be discussed subsequently in the examples as the concepts arise.

![From https://www.geospatialworld.net/blogs/difference-between-ai%EF%BB%BF-machine-learning-and-deep-learning/](https://geospatialmedia.s3.amazonaws.com/wp-content/uploads/2017/05/AAEAAQAAAAAAAAhPAAAAJDlkMWMwNTA1LTZkZjUtNDA5MS1hYT.jpg)


## What is heliophysics?
Heliophysics is a term that encompasses a lot. In short, it refers to the physics of the Sun and how its light and particles interact with everything in the solar system. Our focus tends toward interactions with planetary atmospheres and magnetospheres. This is critically important for us on Earth because _space weather_ can adversely impact many of the technologies we rely on every day. For example, high-energy particles can damage GPS satellites. Farming, radio communications, and airlines all depend on high-precision Guidance and Navigation Satellite Systems (GNSS) like GPS. In fact, all satellites are vulnerable to solar storms, including communications satellites. Space weather also affects avionics, submaries, power grids, and astronauts. There are myriad consequences to severe space weather as detailed in a [2008 National Research Council Report](https://www.nap.edu/catalog/12507/severe-space-weather-events-understanding-societal-and-economic-impacts-a). Fortunately, just as with terrestrial weather, accurate forecasts of space weather allow us to take measures to mitigate these impacts. We hope your ears perked up at the mention of forecasting, which is just a more probabilistic term for _prediction_. 

But heliophysics isn't all about making better space weather forecasts. It's science. Many of us are in it for the joy of discovery and for contributing to humanity's understanding of nature. Not only is it important for understanding the history and future of the solar system, but it is a microscope on the only solar system we know is capable of evolving and harboring life. How does the Earth's magentic field rapidly reconfigure itself during a solar storm? And how do these storms impact the ionosphere? What's the composition of the solar atmosphere? And how hot is it, exactly? How do auroras behave on other planets? What if the Sun were smaller and dimmer? What if it was much more active -- sending off even bigger disruptive eruptions and more often? How do storms on Sun-like stars in other solar systems affect extra-solar planets? Our ever deepening understanding of heliophysics informs our determination of whether these planets are potentially habitable. And in exchange, it proves us with context for our planetary relationship with our star and that most tantalizing question: _Are we alone?_






