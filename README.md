# Trailblazer

Hi! This is the official repository of *Trailblazer* paper "*Large Language Models Enable Generalizable Policy Design in Network Optimization*" (coming soon).

**What is Trailblazer?** 😄

Trailblazer is the pioneering framework that design and deploy network optimization policies with LLMs to achieve stronger generalization. 

Through extensive experiments on two representative networking tasks of broad social and industrial importance-adaptive bitrate streaming (ABR) and cluster job scheduling (CJS)-we show that Trailblazer powered by a single LLM significantly outperforms state-of-the-art baselines, demonstrating stronger cross-task and cross-environment generalization. 

To validate its real-world applicability, we deploy Trailblazer in Douyin’s congestion control (CC) service for large-scale online A/B tests for three weeks, serving 150,000+ users across 100+ cities and accumulating over 1,200 days of video playback time. Results show that Trailblazer outperformed VICC, a strong and mature baseline currently used by Douyin, across all key industrial performance metrics. 

> *Trailblazer (开拓者)* signifies our goal of forging the first path in LLM-powered network systems, establishing a foundational framework for both academia and industry to advance the integration of LLMs into real-world network services. 😊

**What does this repo provide?** 😄

This repository contains the implementation of Trailblazer for the ABR and CJS tasks. Please note that due to Douyin’s data security requirements, we are unable to release the code and data for the CC task deployed in production.

