// auteur: T.Dargent
// date: 16/05/2010
// copyright Thales Alénia Space 2020
// Cour MAM5 Satellite
// Fichier de TD
// Prise d'excentricité sous l'effet de la pression de radiation

clear


// 1/ déclaration des données du problème
/////////////////////////////////////////
// mu_boddy = 3.9860064E+14
// R terre = 6378140
// 1 jour en s = 86400
// I année en j = 365,25



m0 = 2500;          // Masse satellite à t0 en kg
Cp = 1.3;           // on suppose Ca=0.7, Cs = 0.3 et Cd=0 donc Cp=Ca +2*Cs
S  = 40;            // Surface Panneau solaire
F  = Cp*S*1366/3e8; // Pression de radiation en N
g0 = 9.80665;       // Gravité terrestre en ms^-2
mu_boddy    = 3.9860064E+14;  // constante gravitationnel de la terre en m^3s^-2
T= 86400*(360/(360+360/365.25));
a = ((T/2/%pi)^2*mu_boddy)^(1/3);

// Paramètre de Orbite initiale à t0²
x0 = a; //ma réponse (correcte)
y0 = 0; //ma réponse (correcte)
vx0= 0; //ma réponse (correcte)
vy0= sqrt(mu_boddy/a);




// 2/ Choix des unités de normalisation : 
/////////////////////////////////////////

DU= a;   // distance demi grand axe de l'orbite GEO
VU= sqrt(mu_boddy/a);   // vitesse de l'orbite GEO
TU= DU/VU;   // temps
MU= m0;   // masse du satellite à t=0
FU= MU*DU/((TU)*(TU));   // force


// 3/ Décrire la dynamique du système
//////////////////////////////////////////////////////////

// dx=dynsat(t,x)
// dynsat dynamique satellite 
// t le temps
// etat l'état du système dynamique vecteur de dimension 6
// etat = [x;y;vx;vy]
// detat dérivé par rapport au temps de etat
// les paramètres de la dynamique sont donnés par la structure param 
// structure de paramètres du satellite contenant au moins:
// param.mu_boddy constante gravitationnel du corps central
// param.T poussée du moteur
// param.m0 la masse du satellite

function detat=dynsat(t,etat)

// paramètre du corps central, force perturbatrice et satellite
mu     = param.mu_boddy;  // constante gravitationnel
F      = param.F;         // Poussée moteur
m0     = param.m0;        // Masse satellite à t
w_terre = param.w_terre;  // Vitesse de rotation de la terre autour du soleil

// variables d'état
x = etat(1);
y = etat(2);
vx = etat(3);
vy = etat(4);
r = (x^2+y^2)^(1/2);
// Dynamique de l'état
dx=vx;
dy=vy;
dvx=(-mu/(r*r*r))*etat(1)-(F*cos(w_terre)/m0);
dvy=(-mu/(r*r*r))*etat(2)-(F*cos(w_terre)/m0);
detat=[ dx,
dy,
dvx,
dvy
       ]; 
endfunction



// structure de paramètres utilisé pour dynsat(t,x)
// le calcul est fait en utilisant des parametres en unité normalisé
param.mu_boddy  = mu_boddy/DU^3*TU^2;     // constante gravitationnel 1 en Unité normalisée
param.F         = F/FU;  // Poussée moteur en unité normalisée
param.m0        = m0/MU;     // masse satellite à t0 en unité normalisée
param.w_terre   = 2*%pi/(365.25*86400/TU);     // rotation de la terre autour du soleil en unité normalisée


y0   = [x0/DU,y0/DU,vx0/VU,vy0/VU];
t0=0;
t=[0:T/20:T*4]/TU;
rtol=1e-12;
atol=1e-12;
y= ode("rk",y0,t0,t,rtol,atol,dynsat);  // intégration de la dynamique

r=sqrt(y(1,:).*y(1,:)+y(2,:).*y(2,:));
v=sqrt(y(3,:).*y(3,:)+y(4,:).*y(4,:));

figure(1);
plot(t/2/%pi,r-1)
set(gca(),"grid",[1 1])
xlabel("temps en nombre de tour", "fontsize", 2)
ylabel("r/a(t=0)-1","fontsize", 2)

//Calcul de l'exentricité à partir des formules du mouvement képleriens
// calcul de du moment cynetique C=r^v
C=(y(1,:).*y(4,:)-y(2,:).*y(3,:)); // C=sqrt(mu*p)
p=C.^2./param.mu_boddy;
e_cos_nu = p./r-1; // r=p/(1+e*cos(nu))
// calcul du produit scalaire r.v r.v = r*v*sin(phi) et sin(phi) = e*sqrt(mu/p)*sin(nu)/v
rv=y(1,:).*y(3,:)+y(2,:).*y(4,:);
e_sin_nu = rv./(r.*sqrt(param.mu_boddy./p)); // e*sin(nu)
e=sqrt(e_cos_nu.^2+e_sin_nu.^2)
a=p./(1-e.^2)

figure(2);
plot(t/2/%pi,e)
set(gca(),"grid",[1 1]);
xlabel("temps en nombre de tour", "fontsize", 2);
ylabel("e(t)","fontsize", 2);

figure(3);
plot(t/2/%pi,a-1);
set(gca(),"grid",[1 1]);
xlabel("temps en nombre de tour", "fontsize", 2);
ylabel("a/a(t=0)-1","fontsize", 2);


