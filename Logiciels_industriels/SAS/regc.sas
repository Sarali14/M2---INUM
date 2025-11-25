libname stat1 '/export/viya/homes/christine.malot@univ-cotedazur.fr/donnees/';

ods graphics;

proc glm data=STAT1.ameshousing3 plots=diagnostics;
	class Heating_QC;
	model SalePrice=Heating_QC;
	means Heating_QC / hovtest=levene;
	title "One-Way ANOVA with Heating Quality as Predictor";
run;
quit;

ods graphics;
ods select lsmeans diff diffplot controlplot;

proc glm data=STAT1.ameshousing3 plots(only)=(diffplot(center) controlplot);
	class Heating_QC;
	model SalePrice=Heating_QC;
	lsmeans Heating_QC / pdiff=all adjust=tukey;
	title "Post-Hoc Analysis of ANOVA - Heating Quality as Predictor";
run;
quit;

ods graphics off; 
proc means data=STAT1.ameshousing3 
mean std var nway; 
class Season_Sold Heating_QC; 
var SalePrice; 
title 'Selected Descriptive Statistics'; 
run;

proc sgplot data=STAT1.ameshousing3; 
vline Season_Sold / group=Heating_QC 
stat=mean 
response=SalePrice 
markers; 
run;

ods graphics on; 
proc glm data=STAT1.ameshousing3 order=internal; 
class Season_Sold Heating_QC; 
model SalePrice=Heating_QC Season_Sold; 
lsmeans Season_Sold / diff adjust=tukey; 
title "Model with Heating Quality and Season as Predictors"; 
run;


ods graphics / imagemap=on; 
proc glm data=ST141.AMESHOUSING3; 
class Heating_QC Season_Sold; 
model SalePrice=Heating_QC Season_Sold Heating_QC*Season_Sold / ss1 ss3; 
quit;


ods graphics on; 
proc glm data=STAT1.ameshousing3 
order=internal 
plots(only)=intplot; 
class Season_Sold Heating_QC; 
model SalePrice=Heating_QC Season_Sold Heating_QC*Season_Sold; 
lsmeans Heating_QC*Season_Sold / diff slice=Heating_QC; 
store out=interact; 
title "Model with Heating Quality and Season as Interacting "
"Predictors"; 
run;

proc plm restore=interact plots=all; 
slice Heating_QC*Season_Sold / sliceby=Heating_QC adjust=tukey; 
effectplot interaction(sliceby=Heating_QC) / clm; 
run;



