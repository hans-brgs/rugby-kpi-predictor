Kpi selections : 
- carries
- metres made
- defenders beaten
- offloads
- passes
- tackles
- missed tackles
- turnovers conceded
- kicks from hand
- clean breaks
- turnovers won
- lineouts won
- lineouts lost
- scrums won
- scrums lost
- rucks won
- rucks lost
- penalties conceded
- free kicks
- scrum penalties
- lineout penalties
- tackle/ruck/maul penalties
- general play penalties
- control penalties
- yellow cards
- red cards
- home/away status.

Based on : [Performance indicators associated with match outcome within the United Rugby Championship, A. Scott & al. 2022](https://www.sciencedirect.com/science/article/pii/S1440244022004972)


Kpi selections : 
- passes
- runs OR carriesMetres
- carriesCrossedGainLine
- carriesNotMadeGainLine
- offload
- cleanBreaks
- defendersBeaten
- kicksFromHand OR KickFromHandMetres
- kickPossessionLost
- kickPossessionRetained
- scrumsWon
- scrumsLost
- scrumsLost
- lineoutsWon
- lineoutsLost
- tackles
- missedTackles
- penaltiesConceded
- totalFreeKicksConceded
- redCards
- yellowCards
- rucksLost
- rucksWon
- maulsWon OR maulingMetres
- maulsLost
- collectionFailed
- collectionSuccess
- restartsWon
- restartsLost
- turnoverWon
- turnoversConceded OR (turnoverOppHalf + turnoverOwnHalf)
- pcPossessionFirst
- pcPossessionSecond
- pcTerritoryFirst
- pcTerritorySecond
- attackingEventsZoneA
- attackingEventsZoneB
- attackingEventsZoneC
- attackingEventsZoneD
- ballWonZoneA
- ballWonZoneB
- ballWonZoneC
- ballWonZoneD
- isHome

# Random Forest Analysis Road Map
- Effectuez une analyse exploratoire des données pour identifier les corrélations et les variables à faible variance.
- Appliquez une méthode de sélection de variables (ex: Lasso, Boruta) pour réduire le nombre d'indicateurs.
- Validez la sélection avec des experts du rugby.
- Construisez plusieurs modèles Random Forest avec différents sous-ensembles de variables (10-20 max).
- Comparez les performances des modèles en utilisant la validation croisée.
- Choisissez le modèle final offrant le meilleur compromis entre performance et parcimonie.

# Choix du selection des variables 

- Exlusions des données sur les essaies et la transformation :
  - ***Redondance*** : Ma variable cible (à prédire, différence de point) inclus déja indirectement ces deux variables.
  - ***Masquage d'informations*** : On sait déjà que ces deux variables sont fortement corrélé avec l'issue du match, donc (1) ca n'apporte pas d'informations supplémentaire de les inclure. (2) l'inclusion de ces paramètres pourraient forcer le modèle à se focaliser sur ces variables évidentes au détriment d'autres facteurs plus subtils, mais potentiellement plus instructifs sur les conditions qui mènent à la victoire.
> [!NOTE]
> Une alternative serait de réaliser une analyse en deux étapes.
> Analyse 01, modèle sans ces variables. Analyse 02, modèle avec inclusions de ces variables.
> Cela pourrait permettre de comprendre quelles autres variables le second modèle utilises pour "remplacer" l'information des essais et des transformations.

