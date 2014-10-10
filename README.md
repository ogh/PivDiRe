PivDiRe
=======

Dimensionality reduction that allows you to assign weights for the importance of each point

Idea
----

In order to plot high dimensional spaces it's necessary to transform them down to 2D (or maybe 3D with super cool displays). When perform this dimensionality reduction on normal, real word data, that process usually causes a loss of information since there usually is no compact 2D representation of points from a high dimensional space.
Standard methods like PCA or [tSNE](http://homepage.tudelft.nl/19j49/t-SNE.html) treat every point as equally important.
The provided python function allows you to specify a number of _pivot points_ that constitute some kind of landmark in your space. Preserving the distance from those pivot points to any other point is then regarded as more important than any normal point-to-point distance.

Example Scenario
----------------

You retrieve the nearest neighbors for 3 different query points from a set of vectors. You would like to plot each query point and all the retrieved results in the same image.
For most examples that I have encountered, standard methods will give you quite poor results. Most of the time you will either see a very regular pattern or the results being clustered very closely to their cluster centers (queries) with a lot of space between those clusters.
Since Query-to-point distancas are weighed more than normal point-to-point distances in the proposed method, the results are forced to leave their local cluster to preserve the distances to all the queries.


You can see an example in the iPython notebook that I provided.


Details
-------
Details on the method will follow soon.
