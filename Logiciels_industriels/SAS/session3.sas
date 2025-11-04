***************************************************************;
*  Si l'on considère ce code, dire :                          *;
*    Combien il y ad'étapes dans ce programme.                *;
*    Combien il y a d'énoncés dans la procédure print         *;
*    Combien il y a d'énoncés globaux dans le programme       *;
*    Combien d'observations sont lues dans la procédure print *;
***************************************************************;

data mycars;
	set sashelp.cars;
	AvgMPG=mean(mpg_city, mpg_highway);
run;

title "Cars with Average MPG Over 35";
proc print data=mycars;
	var make model type avgmpg;
	where AvgMPG > 35;
run;

title "Average MPG by Car Type";
proc means data=mycars mean min max maxdec=1;
	var avgmpg;
	class type;
run;

title;

********************************************************;
*    Formater ce code afin d'avoir une visulisation.   *;
*       plus sympathique.                              *;  
*    Exécuter le programme et identifier pourquoi il   *;                  *;
*       ne s'exécute pas. Procéder aux corrections en  *;
*       ajoutant un commentaire sur les changements    *;
*    Identifir le nombre de lignes dans canadashoes    *;
*    Où canadashoes a été créée?                       *;
*    Procéder aux modifications afin que canadashoes   *;
*       soit créée de manière permanente dans une      *;
*       appelée session3                               *;
********************************************************;

data canadashoes; set sashelp.shoes;
	where region="Canada;
	Profit=Sales-Returns;run;

prc print data=canadashoes;run;



