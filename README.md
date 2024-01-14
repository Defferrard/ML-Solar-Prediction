# Mini Projet Machine Learning

## Analyse du Dataset

Le dataset est divisé en un [training set](training.csv) de **17855 entrées labelisées** et
un [test set](test_students.csv) de **2304 entrées non-labelisées**.  
Ces datasets contiennent **14 features** :

| Name                | Id                 | min                     | max                     | Commentaires         |
|---------------------|--------------------|-------------------------|-------------------------|----------------------|
| Date                | ``Date``           | ``2022-01-08 02:00:00`` | ``2022-12-10 01:55:00`` |                      |
| Température         | ``temp``           | ``4.944444444444444``   | ``32.05555555555556``   |                      |
| Humidité            | ``humidity``       | ``25.83``               | ``97.01``               |                      |
| Précipitation       | ``precip``         | ``0.0``                 | ``0.37``                |                      |
| Neige               | ``snow``           | ``0.0``                 | ``0.0``                 | Inutile              |
| Profondeur de neige | ``snowdepth``      | ``0.0``                 | ``0.0``                 | Inutile              |
| Vitesse du vent     | ``windspeed``      | ``0.0``                 | ``18.3``                |                      |
| Direction du vent   | ``winddir``        | ``0.0``                 | ``360.0``               |                      |
| Visibilité          | ``visibility``     | ``1.2``                 | ``34.2``                | Manque bcp de valeur |
| Couverture          | ``cloudcover``     | ``0.0``                 | ``100.0``               |                      |
| Radiation solaire   | ``solarradiation`` | ``0.0``                 | ``1076.0``              |                      |
| Energie solaire     | ``solarenergy``    | ``0.0``                 | ``3.9``                 |                      |
| Index UV            | ``uvindex``        | ``0.0``                 | ``10.0``                |                      |
| Conditions          | ``conditions``     | -                       | -                       | Valeurs textuelles   |

- Domaine de ``conditions`` : ``['Clear' 'Partially cloudy' 'Rain, Partially cloudy' 'Rain' nan 'Overcast'
'Rain, Overcast']``

Le but est de déterminer la production solaire ``solar_production``.
- Minimum : ``0.0``
- Maximum : ``3.3061636363636366``
- Moyenne : ``0.602948998655914``
- Variance : ``0.8435500460472392``

### Questions
- Est-ce que toutes les caractéristiques sont-elles pertinentes ?
Non, ``snow`` et ``snowdepth`` étant toujours à ``0``, ils ne seront pas utiles.
On ne peut pas être sûr que ``windir`` soit utile également.
``visibility`` ayant beaucoup de valeur NaN, il sera pas forcément utile.
- Y a-t-il des valeurs manquantes parmi les échantillons ?
Oui, beaucoup, surtout dans ``visibility``
- Y a-t-il une caractéristique doit-elle être normalisée ?
- Pouvez-vous combiner des fonctionnalités, les modifier ou en créer de nouvelles ?
- Une corrélation ?
- Comment les données sont-elles réparties dans chaque classe ?
- Des valeurs aberrantes ?

## Choix des algorithmes