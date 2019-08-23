---
title: "Text mining for finding the best place to pursue medical research"
layout: post
date: 2016-04-11
<!-- tag: jekyll -->
image: https://raw.githubusercontent.com/AlanFermat/Text-mining-for-medical-research/master/data-mining.jpg
headerImage: true
projects: true
hidden: true # don't count this post in blog pagination
description: "A guessing game based on Twitter API."
category: project
author: Alan Yu
comments: true
externalLink: false

---

![Screenshot](https://raw.githubusercontent.com/AlanFermat/Text-mining-for-medical-research/master/wordcloud.jpg)

## Text mining for finding the best place pursuing medical research

This is an open-end project that I worked with a team in a Statistic class at Rice University.

### Where is the best place for medical research?

We explore where the best places for health research and innovation are, and how this relates to the rates of specific illnesses.

“Where” is meant to be general and refers not only to specific locations, such as cities/states, but also specific diseases, such as obesity, diabetes, and other characteristics. We then rank the 11 top cities for health research and innovation.

In order to figure out where the best places are, we first analyze funding on three dimensions: type of disease, time, and location.

Based on this high-level analysis, we find location is the most important factor. We deep dive into location, specifically city-level analysis.

We breakdown the importance into three categories, which we call: access to talent, access to resources, and access to funding. Based on these three dimensions, we rank the top 11 cities. Based on our ranking system. New York is the top city, while Seattle is the bottom of the 11 cities.

The results are presented beautifully via _ggplot_ and _wordcloud_ using R!

---

[Check it out](https://github.com/AlanFermat/Text-mining-for-medical-research) here.
If you need some help, just [tell me](https://github.com/AlanFermat/Text-mining-for-medical-research/issues).
