# Project overview 

The project aims to apply supervised learning algorithms for hydrogen energy generation scenarios.
Since I have limited access to public data related to the hydrogen-related data, I managed to generate a synthetic data for hydrogen generation. 
In short, the first step is to build a synthetic data set for the  metal-based hydrogen production process using water-splitting reactions. 
The folder **Data Generator** is specifically dedicated for this purpose. 
Before diving deep into the coding-related aspects of the dataset, it is necessary to understand the real-world aspects of the hydrogen production. 

`2Al + 6H₂O → 2Al(OH)₃ + 3H₂ _______________(1)`

`Mg + 2H₂O → Mg(OH)₂ + H₂ __________________(2)`

`Zn + H₂O → ZnO + H₂ _______________________(3)`

# Factors influencing the cost of hydrogen production based on metal hydrolysis

The cost modeling accounts for both raw materials and operational factors.

## Raw metal cost

The cost of the metal used for hydrogen production is one of the major costs associated with the process. 
Based on the report from the Department of Energy in the USA, approximately 9kg of aluminium is used to produce one kg of hydrogen (https://www1.eere.energy.gov/hydrogenandfuelcells/pdfs/aluminium_water_hydrogen.pdf).
During the reaction process with aluminium oxide, formation occurs, thereby hindering the steady production of hydrogen. 
Therefore, some extra materials like Indium or Gallium can be added to stop the oxide formation; this could further increase the raw material price pertaining only to the metal. 
In cost models, raw metal cost is calculated based on the stochiometric metal requirement per kg of H2. Recycling metal can reduce costs. 

## Energy consumption 

Energy inputs are needed either to initiate or maintain the reaction. Moreover, the energy inputs are also used for the regeneration process for the reuse of metal. 
For example, in the aluminum-based hydrolysis process, the energy required for electrolysis is roughly 620 MJ (electrical energy).
In summary, energy consumption costs cover any electricity for electrolysis, heat for reactors, pumping power, and even chilling or compression if hydrogen needs purification – all based on the energy and thermodynamic requirements of the process.

## Catalysts and chemical additives
