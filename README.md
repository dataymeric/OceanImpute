# Deep Learning for Data Imputation in Oceanography

This repository contains code from my Master's research internship, co-supervised at [ISIR](https://www.isir.upmc.fr) and [LOCEAN](https://www.locean.ipsl.fr) labs (Sorbonne Université). Remote sensing provides essential data for monitoring ocean color and phytoplankton, which are important indicators of marine ecosystem health. However, missing data is a common issue in these observations, and addressing it is necessary to gain a complete understanding of ocean dynamics.

This project presents a transformer-based model designed to impute missing values in variables such as sea surface temperature, chlorophyll-a, and phytoplankton size classes. The model captures spatial, temporal, and multivariate correlations in 3D oceanographic data using self-attention mechanisms. The ability to handle sequences and multivariate data makes this approach a promising tool for oceanographic research.

![Model architecture](https://cdn.discordapp.com/attachments/1079151811388260352/1316444256939933776/architecture.png?ex=675b11c1&is=6759c041&hm=483313c2584514988c6d12fbb4122fbe11771f1dc1e387fbe9c357b190d2c66c&)

You can find the [full report](https://drive.google.com/file/d/1psOIKY2l0VzEtAnabk16bcPOwmkGBQ3x/view) and [presentation slides](https://drive.google.com/file/d/1ar_p49QlclPMM0aLz1v1n1aMRtrFDfrq/view) for further details.