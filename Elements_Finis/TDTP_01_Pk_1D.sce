clear
clf
// ------------------------------------------------------------
// Ici on construit la les deux fonctions de base P1 en 1D
// en un point x ; et pour un element défini par x1 et x2
// -----------------------
// 
//-------------------------------------------------------------
function out=Lambda_of_P1(x, x1, x2)
   out(1) = (x - x2)/(x1-x2)
   out(2) = (x - x1)/(x2-x1)
endfunction

//--------------------------------
// Les gradients des fonctions de Base
//--------------------------------
function [G] = MGrad_Lambda_of_P1(x1, x2)  
    G(1) = 1.0/(x1-x2)
    G(2) = 1.0/(x2-x1)
endfunction
//---------------------------------
// function second membre de problème.
//--------------------------------------
function  out = Rhs_Function(x,rhs)
    select rhs
    case 1 then 
        out =  x 
    case 2 then
        out =  x - 1
    else
        out =  x
    end
endfunction

// =====================================================
function [Bf_at_Xg, Grad_Bf_at_Xg] = Bf_et_Grad(Nve,xe1,xe2,Xsi)

    Grad_Xsi = MGrad_Lambda_of_P1(xe1, xe2) 
    
        Xsi1 = Xsi(1,1:Ng)   ;     Xsi2 = Xsi(2,1:Ng)

    select Nve 

    case 2 // P1-Lagrange 1D
        // ------------------------------------------------------
        Bf_at_Xg(1,1:Ng) =  Xsi1 
        Bf_at_Xg(2,1:Ng) =  Xsi2; 

        Grad_Bf_at_Xg(1,1:Ng) = Grad_Xsi(1)
        Grad_Bf_at_Xg(2,1:Ng) = Grad_Xsi(2)
        disp("P1-Lagrange 1D" )
    case 3 // P2-Lagrange 1D
        // ------------------------------------------------------

        Bf_at_Xg(1,1:Ng) =  Xsi1.*( 2.0*Xsi1 - 1)   
        Bf_at_Xg(2,1:Ng) =  Xsi2.*( 2.0*Xsi2 - 1)
        Bf_at_Xg(3,1:Ng) =  4.0*Xsi1.*Xsi2

        Grad_Bf_at_Xg(1,1:Ng) =    Grad_Xsi(1)*(4.0*Xsi1 - 1)
        Grad_Bf_at_Xg(2,1:Ng) =    Grad_Xsi(2)*(4.0*Xsi2 - 1) 
        Grad_Bf_at_Xg(3,1:Ng) = -4*Grad_Xsi(2)*(2.0*Xsi2 - 1)
        disp("P2-Lagrange 1D" )
    case 4 // P3-Lagrange 1D
        // ------------------------------------------------------
        disp("P3-Lagrange 1D" )
        L1 = Xsi(1,1:Ng)   ;         L2 = Xsi(2,1:Ng)
        
        Bf_at_Xg(1,1:Ng) =   L1.*(3*L1 -1).*(3*L1 - 2)/2 
        Bf_at_Xg(2,1:Ng) =   L2.*(3*L2 -1).*(3*L2 - 2)/2
        Bf_at_Xg(3,1:Ng) =        9*L1.*L2.*(3*L1 - 2)/2
        Bf_at_Xg(4,1:Ng) =        9*L1.*L2.*(3*L2 - 2)/2 


        dL1 = Grad_Xsi(1)  ;        dL2 = Grad_Xsi(2)

        Grad_Bf_at_Xg(1,1:Ng) = dL1.*(3*L1 -1).*(3*L1 - 2)/2 ...
                                  + L1.*(3*dL1  ).*(3*L1 - 2)/2 ...
                                  + L1.*(3*L1 -1).*(3*dL1   )/2   
        Grad_Bf_at_Xg(2,1:Ng) =  dL2.*(3*L2 -1).*(3*L2 - 2)/2 ...
                                  + L2.*(3*dL2  ).*(3*L2  - 2)/2 ...
                                  + L2.*(3*L2 -1).*(3*dL2    )/2     
        Grad_Bf_at_Xg(3,1:Ng) = 9*dL1.*L2 .*(3*L1 - 1)/2 ...
                                  + 9*L1.*dL2.*(3*L1 - 1)/2 ...
                                  + 9*L1.*L2 .*(3*dL1   )/2  
        Grad_Bf_at_Xg(4,1:Ng) = 9*dL1.*L2 .*(3*L2 - 1)/2 ...
                                  + 9*L1.*dL2.*(3*L2 - 1)/2 ...
                                  + 9*L1.*L2 .*(3*dL2   )/2  
        // -------------------------------------------------------

    else // P1-Lagrange 1D (par défaut)
        // ------------------------------------------------------
        Bf_at_Xg(1,1:Ng) =  Xsi(1,1:Ng)   
        Bf_at_Xg(2,1:Ng) =  1.0 - Xsi(1, 1:Ng); 

        Grad_Bf_at_Xg(1,1:Ng) = Grad_Xsi(1)
        Grad_Bf_at_Xg(2,1:Ng) = Grad_Xsi(2)

    end

endfunction
// =====================================================

//  donnees pour integration avec deux points de Gauss
// -------------------------------------------------
function [Poid, Xsi, Ngo]= IntegrationNum(Ngi)
    Ngo  = Ngi
    
    select Ngi
    case 1 then// integration numérique avec UN points de Gauss.
        Poid(1)      = 1.0;
        Xsi(1,1)     = 0.5;
    case 2 then// integration numérique avec DEUX points de Gauss.
        Poid(1:2)   = 0.5;
        Xsi(1,1)    = 0.5*( 1 - 1.0/sqrt(3)) ;
        Xsi(1,2)    = 0.5*( 1 + 1.0/sqrt(3)) ;

    case 3 then// integration numérique avec TROIS points de Gauss.
        Poid(1)  = 5.0/18.0 ;
        Poid(2)  = 8.0/18.0 ;
        Poid(3)  = 5.0/18.0 ;

        Xsi(1, 1) = 0.5 * ( 1.0 - sqrt(3.0/5.0) ) ;
        Xsi(1, 2) = 0.5   ;
        Xsi(1, 3) = 0.5 * ( 1.0 + sqrt(3.0/5.0) ) ;
 
    else
        // une autre alternative
        Ngo       = 3

              
        Poid(1)  = 1.0/6.0 ;
        Poid(2)  = 4.0/6.0 ;
        Poid(3)  = 1.0/6.0 ;

        Xsi(1, 1) = 0.0   ;
        Xsi(1, 2) = 0.5   ;
        Xsi(1, 3) = 1.0   ;

    end  
    // les lambda 2
    // =======================================
    Xsi(2, 1:Ngo)  = 1.0 - Xsi(1,1:Ngo);

endfunction

// fonctions de base aux points de Gauss
// comme les points de Gauss sont donnes dans [0, 1]
// les fonctions et les points de Gauss coincident
// ====================================================
function [Ae, Be]= Ae_et_Be_Num(NpG,Pk,xe1,xe2,Mu,Reac,rhs)
    
    [W_g, Xsi_g, Ng]     = IntegrationNum(NpG)

    // =============================
    
    dx  =  abs(xe2-xe1);
    Nve = Pk + 1;
    // Initialisations à zéro de la matrice et du second membre
    // ---------------------------------------------------------
    Ae  = zeros(Nve, Nve);
    Be  = zeros(Nve,1);

    // Fonctions de base et Gradients
    [Bf_at_Xg, GradBf_at_Xg] = Bf_et_Grad(Nve,xe1,xe2,Xsi_g)

    // boucles sur les points de Gauss
    // ===============================
    for gp = 1:Ng

        xg = Xsi_g(1,gp)*xe1 + Xsi_g(2,gp)*xe2 ;

        // -----------------------------------------------------
        // Second membre local et   Matrice Locale
        //        assemblage
        // -----------------------

        // premiere boucle sur le ddl locaux
        // ==================================
        
        for i = 1: Nve
            Bf_i    =     Bf_at_Xg(i,gp);
            GBf_i   = GradBf_at_Xg(i,gp);

            fg      = Rhs_Function(xg,rhs);

            bi_g    = fg * Bf_i ;

            Be(i)   =  Be(i) + dx * W_g(gp) * bi_g;
            // seconde boucle sur les ddl locaux
            // =================================
            for j = 1:Nve
                Bf_j    =     Bf_at_Xg(j, gp);
                GBf_j   = GradBf_at_Xg(j, gp);

                aij_g   = Mu*GBf_i*GBf_j + Reac*Bf_i*Bf_j;

                Ae(i, j)= Ae(i, j) + dx * W_g(gp) * aij_g;
            end

        end
    end
endfunction

function [Aex, Bex] = Exo_2_1_1(xe1,xe2)
// Exo 2.1.1
// ------------
// Matrice locale analytique
// ********************************************
Aex=[  1.0/abs(xe1-xe2) + abs(xe1-xe2)/3.0, ...
      -1.0/abs(xe1-xe2) + abs(xe1-xe2)/6.0; ...
      -1.0/abs(xe1-xe2) + abs(xe1-xe2)/6.0, ...
       1.0/abs(xe1-xe2) + abs(xe1-xe2)/3.0] 
       
// Second membre local analytique
// ********************************************            
Bex=[abs(xe1-xe2)*(2*xe1+xe2)/6.0;... 
     abs(xe1-xe2)*(xe1+2*xe2)/6.0 ]
     
endfunction 


function [Aex, Bex] = Exo_2_1_2(xe1,xe2)
// Exo 2.1.1
// ------------
// Matrice locale analytique
// ********************************************
dx = abs(xe1-xe2)
Aex=[  7.0/(3*dx),  1.0/(3*dx) -8.0/(3*dx) ; ...
       1.0/(3*dx),  7.0/(3*dx) -8.0/(3*dx) ; ...
      -8.0/(3*dx), -8.0/(3*dx) 16.0/(3*dx) 
      ] 
       
// Second membre local analytique
// ********************************************            
Bex=[dx*(xe1-1)/6.0; dx*(xe2-1)/6.0;... 
                     dx*(xe1+xe2-2)/3.0]
     
endfunction 



disp("******************************************************")
disp("Exo 2.1.1 ")
Pk=1
xe1      = 2.0
xe2      = 3.0
Mu       = 1.0
alpha    = 1.0
Np_Gauss = 3
for Np_Gauss =  1:3
rhs=1
[Afe, Bfe]= Ae_et_Be_Num(Np_Gauss,Pk,xe1,xe2,Mu,alpha,rhs)

[Ae_P1,Be_P1] = Exo_2_1_1(2.0,3.0)

printf("   Nombre de points de Gauss = %2i\n",Np_Gauss);
printf("   -------------------------------\n");
disp("Ae_analytique - Ae_numérique")
disp([Ae_P1-Afe])
disp("Be_analytique - Be_numérique")
disp([Be_P1-Bfe])
disp("------------------------------------------------------")
end 
disp("******************************************************")
disp("Exo 2.1.2 ")
Pk=2
xe1      = 2.0
xe2      = 3.0
Mu       = 1.0
alpha    = 0.0
rhs      = 2
for Np_Gauss =  1:3
[Afe, Bfe]= Ae_et_Be_Num(Np_Gauss,Pk,xe1,xe2,Mu,alpha,rhs)

[Ae_P2,Be_P2] = Exo_2_1_2(2.0,3.0)

printf("   Nombre de points de Gauss = %2i\n",Np_Gauss);
disp("Ae_analytique - Ae_numérique")
disp([Ae_P2-Afe])
disp("Be_analytique - Be_numérique")
disp([Be_P2-Bfe])
disp("------------------------------------------------------")
end 
