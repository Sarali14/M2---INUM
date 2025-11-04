
*Write a PROC CONTENTS step reading storm_summary.sas7bdat;


***********************************************************;                                          *;
*       Après avoir modifier path, exécuter 2 fois le     *;
            programme. Que constatez-vous?                *;
*       Modifier le programme de manière à pouvoir        *;
*           l'exécuter avec succès.                       *;
***********************************************************;

proc import datafile="path/storm_damage.tab" 
			dbms=tab out=storm_damage_tab;
run;


***************************************** ************;                                         *;
*       Modifier dans %LET NA en SP                  *;                                               *;
*       Que remarquez-vous après exécution? Pourquoi?*;
******************************************************;

%let BasinCode=NA;

proc means data=pg1.storm_summary;
	where Basin="&BasinCode";
	var MaxWindMPH MinPressure;
run;

proc freq data=pg1.storm_summary;
	where Basin='&BasinCode';
	tables Type;
run;


***********************************************************;
*       Exécuter uniquement la procédure print.           *;
*         Que notez-vous au sujet des variables Lat, Lon, *;
*         StartDate, et EndDate                           *;
*       Modifier le format de DATE à 7 puis à 11          *;
*          Exécuter à chaque fois et noter les différences*;
*       Exécuter la procédure freq                        *;
*       Ajouter un format MONNAME à startdate et exécuter *;
*          de nouveau la précédure freq. Que notez-vous?  *;
***********************************************************;

proc print data=pg1.storm_summary(obs=20);
	format Lat Lon 4. StartDate EndDate date9.;
run;

proc freq data=pg1.storm_summary order=freq;
	tables StartDate;
run;


***********************************************************;
*       Modifier lib et l'option out afin de créer une    *;
*         base temporaire tempete.                        *;
*       Compléter les instructions where et by de sorte   *;
*         que l'on puisse identifier la tempete de North  *;
*         atlantic avec le vent le plus fort.             *;
***********************************************************;

proc sort data=lib.storm_summary out=;
	where ;
	by ;
run;


***********************************************************;
*       Obtenir une base de sortie tempete_cat5           *;
*         ne contenant que les tempetes de catégories 5   *;
*         ventmax >=156 et postérieure au 1 janv 2020     *;
*         Ne conserver que les colonnes Season, Basin,    *;
*          Name, Type, MaxWindMPH                         *;
***********************************************************;


data out.storm_new;
    set lib.storm_summary;
run;


***********************************************************;                                         *;
*    Ajouter une colonne donnant la durée de la tempete.  *;                            *;
*    En 1980, quelle a été la durée de Agatha last?       *;                         *;
***********************************************************;

data storm_length;
	set lib.storm_summary;
	drop Hem_EW Hem_NS Lat Lon;
run;


***********************************************************;
*       Examiner la table STORM_RANGE et notez qu'il y a  *;
*         4 mesures de vent                               *;                             *;
*       Créer une colonne WindAvg calculant la moyenne    *;
*         de Wind1, Wind2, Wind3, et Wind4.               *;
*       Créer une colonne WindRange donnant l'étendue     *;
*       entre Wind1, Wind2, Wind3, et Wind4.              *;
***********************************************************;

data storm_wingavg;
	set lib.storm_range;
run;


***********************************************************;
*    1) Add a WHERE statement that uses the SUBSTR        *;
*       function to include rows where the second letter  *;
*       of Basin is P (Pacific ocean storms).             *;
*    2) Run the program and view the log and data. How    *;
*       many storms were in the Pacific basin?            *;
***********************************************************;
*  Syntax                                                 *;
*     SUBSTR (char, position, <length>)                   *;
***********************************************************;

data pacific;
	set pg1.storm_summary;
	drop Type Hem_EW Hem_NS MinPressure Lat Lon;
	*Add a WHERE statement that uses the SUBSTR function;
run;


***********************************************************;
*    1) Add the ELSE keyword to test conditions           *;
*       sequentially until a true condition is met.       *;
*    2) Change the final IF-THEN statement to an ELSE     *;
*       statement.                                        *;
*    3) How many storms are in PressureGroup 1?           *;
***********************************************************;

data storm_cat;
	set pg1.storm_summary;
	keep Name Basin MinPressure StartDate PressureGroup;
	*add ELSE keyword and remove final condition;
	if MinPressure=. then PressureGroup=.;
	if MinPressure<=920 then PressureGroup=1;
	if MinPressure>920 then PressureGroup=0;
run;

proc freq data=storm_cat;
	tables PressureGroup;
run;


***********************************************************;
*    1) Run the program and examine the results. Why is   *;
*       Ocean truncated? What value is assigned when      *;
*       Basin='na'?                                       *;
*    2) Modify the program to add a LENGTH statement to   *;
*       declare the name, type, and length of Ocean       *;
*       before the column is created.                     *;
*    3) Add an assignment statement after the KEEP        *;
*       statement to convert Basin to uppercase. Run the  *;
*       program.                                          *;
*    4) Move the LENGTH statement to the end of the DATA  *;
*       step. Run the program. Does it matter where the   *;
*       LENGTH statement is in the DATA step?             *;
***********************************************************;
*  Syntax                                                 *;
*       LENGTH char-column $ length;                      *;
***********************************************************;

data storm_summary2;
	set pg1.storm_summary;
	keep Basin Season Name MaxWindMPH Ocean;
	OceanCode=substr(Basin,2,1);
	if OceanCode="I" then Ocean="Indian";
	else if OceanCode="A" then Ocean="Atlantic";
	else Ocean="Pacific";
run;


**********************************************************;
*    Exécuter le programme. Pourquoi il plante ?         *;
**********************************************************;

data front rear;
    set sashelp.cars;
    if DriveTrain="Front" then do;
        DriveTrain="FWD";
        output front;
    else if DriveTrain='Rear' then do;
        DriveTrain="RWD";
        output rear;
run;*


