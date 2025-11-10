libname stat1 '/export/viya/homes/christine.malot@univ-cotedazur.fr/donnees/';

proc print data=stat1.ameshousing3(obs=10) label;
run;


%let categorical=House_Style Overall_Qual Overall_Cond Year_Built Fireplaces Mo_Sold Yr_Sold Garage_Type_2 Foundation_2 Heating_QC Masonry_Veneer Lot_Shape_2 Central_Air;
%let interval=Saleprice Log_price Gr_liv_area Basement_area garage_area Deck_Porch_Area Lot_Area Age_Sold Bedroom_AbvGr Full_Bathroom Half_Bathroom Total_Bathroom;

proc univariate data=stat1.ameshousing3;
var &interval;
histogram &interval / normal kernel;
inset n mean std / position=ne;
run;

proc freq data=stat1.ameshousing3;
table &categorical / norow nocol plots(only)=(freqplot);
run;


ods graphics; 
proc ttest data=STAT1.ameshousing3 
plots(shownull)=interval H0=135000; 
var SalePrice; 
title "One-Sample t-test testing whether mean SalePrice=$135,000"; 
run;



ods graphics;

proc ttest data=STAT1.ameshousing3 plots(shownull)=interval;
	class Masonry_Veneer;
	var SalePrice;
	title "Two-Sample t-test Comparing Masonry Veneer, No vs. Yes";
run;


proc sgscatter data=STAT1.ameshousing3;
	plot SalePrice*Gr_Liv_Area / reg;
	title "Associations of Above Grade Living Area with Sale Price";
run;


%let interval=Gr_Liv_Area Basement_Area Garage_Area Deck_Porch_Area Lot_Area 
	Age_Sold Bedroom_AbvGr Total_Bathroom;
options nolabel;

proc sgscatter data=STAT1.ameshousing3;
	plot SalePrice*(&interval) / reg;
	title "Associations of Interval Variables with Sale Price";
run;


proc sgplot data=STAT1.ameshousing3;
	vbox SalePrice / category=Central_Air ;
title "Sale Price Differences across Central Air";
run;

%macro box(dsn = , response = , charvar = ); 
%let i = 1 ; %do %while(%scan(&charvar,&i,%str( )) ^= %str()) ;

%let var = %scan(&charvar,&i,%str( )); 
proc sgplot data=&dsn; 
vbox &response / category=&var grouporder=ascending connect=mean; 
title "&response across Levels of &var"; 
run; 

%let i = %eval(&i + 1 ) ; 
%end ; 
%mend box;

%box(dsn = STAT1.ameshousing3, 
response = SalePrice, 
charvar = &categorical);


ods graphics;

proc reg data=STAT1.ameshousing3;
	model SalePrice=Lot_Area;
	title "Simple Regression with Lot_Area as Regressor";
run;

quit;