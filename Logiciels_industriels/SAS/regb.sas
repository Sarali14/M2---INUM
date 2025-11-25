libname stat1 '/export/viya/homes/christine.malot@univ-cotedazur.fr/donnees/';


ods graphics;

proc reg data=STAT1.ameshousing3;
	model SalePrice=Lot_Area;
	title "Simple Regression with Lot_Area as Regressor";
run;

quit;

proc reg data=STAT1.ameshousing3 plots(only)=(QQ RESIDUALBYPREDICTED);
	model SalePrice=Lot_Area;
	output out=res student=studres;
	title "Simple Regression with Lot_Area as Regressor";
run;

proc univariate data=res;
var studres;
histogram studres / normal;
run;
quit;

ods graphics on;

proc reg data=STAT1.ameshousing3;
	model SalePrice=Basement_Area Lot_Area;
	title "Model with Basement Area and Lot Area";
run;


%let interval=Gr_Liv_Area Basement_Area Garage_Area Deck_Porch_Area Lot_Area Age_Sold Bedroom_AbvGr Total_Bathroom ;
ods graphics on;

ods graphics on;

proc reg data=STAT1.ameshousing3 plots(only)=(rsquare adjrsq);
    model SalePrice=&interval / selection= adjrsq;
	title "All Possible Model Selection for SalePrice";
run;

quit;
proc glmselect data=STAT1.ameshousing3 plots=all;
	 model SalePrice=&interval / selection=forward details=steps 
		select=SL slentry=0.05;
	title "Stepwise Model Selection for SalePrice - SL 0.05";
run;

proc glmselect data=STAT1.ameshousing3 plots=all;
	 model SalePrice=&interval / selection=backward details=steps 
		select=SL slstay=0.05;
	title "Stepwise Model Selection for SalePrice - SL 0.05";
run;

proc glmselect data=STAT1.ameshousing3 plots=all;
	 model SalePrice=&interval / selection=stepwise details=steps 
		select=SL slstay=0.05 slentry=0.05;
	title "Stepwise Model Selection for SalePrice - SL 0.05";
run;

proc glmselect data=STAT1.ameshousing3 plots=all;
	 model SalePrice=&interval / selection=lasso;
	title "Lasso Model Selection for SalePrice";
run;

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