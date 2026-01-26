if ~exists("RUN_SIMULATION","local") then
    clear;
end

clf

//---------------
function [out] = Exact(x,y,kx,ky)
    out = sin(x*kx).*cos(y*ky)
endfunction
// =============================================================
// =============================================================

  //
function [out] = f(X,kx,ky,bx,by,cx,cy,alpha)
    x = X(1)
    y = X(2)
    out = (alpha + bx*kx * kx + by*ky * ky) * Exact(x, y, kx, ky)+cx*kx*cos(kx*x)*cos(ky*y)-cy*ky*sin(kx*x)*sin(ky*y)
endfunction
// =============================================================

 //--------------------------------
function [Conv, Reac, Diff] = Composite_Mat(Xg, pb, Lx, Ly, Ne,i, mExp)

    // -------- defaults (optional args) ----------
    // allowed calls:
    //   Composite_Mat(Xg, pb, Lx, Ly, Ne)
    //   Composite_Mat(Xg, pb, Lx, Ly, Ne, i)
    //   Composite_Mat(Xg, pb, Lx, Ly, Ne, i, mExp)
    if argn (2) < 6 then
        i = 0; // by default 
    end
    if argn(2) < 7 then
        mExp = 1;       // default exponent for dh^m
    end

    // -------- base init (always return something) ----------
    Conv = [0.0 ; 0.0];
    Reac = 0.0;
    Diff = diag([0.0; 0.0]);

    // -------- choose problem ----------
    select pb

    case 1 then
        Conv = [0.0 ; 0.0];
        Reac = 1.0;
        Diff = diag([0.0; 0.0]);

    case 2 then
        Conv = [0.0 ; 0.0];
        Reac = 0.0;
        Diff = diag([4.0; 4.0]);

    case 3 then
        Conv = [0.0 ; 0.0];
        Reac = 0.0;
        Diff = diag([1.0; 2.0]);

    case 4 then
        Conv = [0.0 ; 0.0];
        Reac = 0.0;
        Diff = diag([1.0; 0.0]);

    case 5 then
        Conv = [0.0 ; 0.0];
        Reac = 0.0;
        Diff = diag([1e-8; 1.0]);

    case 6 then
        Conv = [0.5 ; 1.0];
        Reac = 0.0;

        if i == 1 then
            dh = sqrt ( Lx * Ly / Ne ) ;
            Diff = diag ([ dh ^ mExp ; dh ^ mExp ]) ;
        else
            Diff = diag ([0.0; 0.0]) ;
        end

    case 7 then
        Conv = [1.0 ; 0.5];
        Reac = -5.0;
        Diff = diag([1.0; 2.0]);

    else
        error(msprintf("Composite_Mat: unknown pb = %d", pb));
    end

endfunction

// =============================================================
// =============================================================

// =============================================================
 
//-----------------------------------------------------------
// Construction d'un maillage stricturé Pk de triangles 
//                   Domaine rectangulaire . 
//--------------------------------------------------------------
function [N_vertives, N_elements, Coor, Nu, LogP, LogE, NuVe] ...
                          = Struct_Pk_Mesh(Nx, Ny, Lx, Ly,pk)
    
    MNx        = Nx + (pk-1)*(Nx - 1) 
    MNy        = Ny + (pk-1)*(Ny - 1) 
    
    N_vertives = MNx*MNy
    N_elements = 2*(Nx-1)*(Ny-1)
    
    dx = Lx/(MNx -1)
    dy = Ly/(MNy -1)
    is = 0
    
    for i=1:MNx
        xi = (i-1)*dx
        for j=1:MNy
            yj = (j-1)*dy
            is = is +1
            Coor(1:2, is)=[xi,yj]'
            LogP(is) = 0
            
            if i==1 then
                LogP(is) = -1
            elseif i==MNx then
                LogP(is) = -2
            elseif j==1 then
                LogP(is) = -10
            elseif j==MNy then
                LogP(is) = -20
            end
        end
    end
    
   ie = 0
    for i=1:Nx-1
        for j=1:Ny-1
            
            is = 1 + pk*(j-1) + (i-1)*pk*MNy ; 
            js = is + pk*MNy ;
            ks = js + pk ;
            ps = is + pk ;
            
            // if( 0 == 0 ) then
                // element du bas 
                // is ---> js  ---> ps ---> is
                // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ie = ie +1
                
                Nu(1:3, ie) = [is  ; js ;  ps] ;
                
                il  = 3 ; 
                iie = 2
                // horizontale  (is --> js)
                // suivant le vecteur (1,0)
                // -----------------------
                for lx=1:pk-1
                    il         = il +1
                    Nu(il, ie) = is + lx*MNy ; 
                end 
                iie = il - pk + 2
                ije = il
                // oblique   (js --> ps ) 
                // suivant le vecteur (-1,1)
                // ----------------------------------
                for lo=1:pk-1
                    il         = il +1
                    Nu(il, ie) = js  + lo - lo*MNy ; 
                end 
                jje= il - pk + 2
                jke=il
                // verticale   (ps --> is)
                // suivant le vecteur (0,-1)
                // -----------------------------
                for ly=1:pk-1
                    il         = il +1
                    Nu(il, ie) = ps - ly ; 
                end 
                kke=il - pk +2
                kie=il
                // -----------------------------
                // ddl dans l'élément du bas
                // ----------------------------
                // -----------------------------
                for lx =1:pk-2
                    for ly = 1:lx
                        il = il +1
                        Nu(il, ie) = is + lx*MNy + ly
                    end
                end
                
               
            
                LogE(ie)    = 0
                            
                // element du haut 
                // js ---> ks ---> ps ---> js
                // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                ie = ie +1 ;
                
                Nu(1:3, ie) = [js ;  ks ; ps] ;
                
                il = 3 ; 
                // verticale  (js --> ks)
                // suivant le vecteur (0,+1)
                // -----------------------------
                for ly=1:pk-1
                    il         = il +1
                    Nu(il, ie) = js + ly ; 
                end 
                // horizontale  (ks --> ps)
                // suivant le vecteur (0,-1)
                // -----------------------------
                for lx=1:pk-1
                    il         = il +1
                    Nu(il, ie) = ks  - lx*MNy ; 
                end 
                // oblique   (ps --> js ) 
                // suivant le vecteur (-1,1)
                // ----------------------------------
                for lo=1:pk-1
                    il         = il +1
                    Nu(il, ie) = ps - lo + lo*MNy; 
                end 
                 
                // -----------------------------
                // ddl dans l'élément du haut
                // ----------------------------
                // -----------------------------
                for ly =1:pk-2
                    for lx = 1:ly
                        il = il +1
                        Nu(il, ie) = js - lx*MNy + ly +1
                    end
                end
                
                LogE(ie)    = 0
        
        end
    end
    
     
                    
   
    if (pk==1) then
        NuVe(1:3,1) = [1;2;3]
        
    elseif (pk==2) then
        NuVe(1:3,1) = [1;4;6]
        NuVe(1:3,2) = [4;5;6]
        NuVe(1:3,3) = [4;2;5]
        NuVe(1:3,4) = [6;5;3]
        
    elseif (pk==3) then
        NuVe(1:3,1) = [1 ; 4; 9]
        NuVe(1:3,2) = [4 ;10; 9]
        NuVe(1:3,3) = [4 ; 5;10]
        NuVe(1:3,4) = [5 ; 6;10]
        NuVe(1:3,5) = [5 ; 2; 6]
        NuVe(1:3,6) = [9 ;10; 8]
        NuVe(1:3,7) = [10; 7; 8]
        NuVe(1:3,8) = [10; 6; 7]
        NuVe(1:3,9) = [8 ; 7; 3]
     end
    

endfunction
//---------------------------------------------------------------
//---------------------------------------------------------------
//
// Fonctions de base Pk-Lagrange pour k= 1, 2 et 3
// --------------------------------------------
   function Phi = Phi_Pk(lmd1, lmd2, pk)
       lmd3 = 1.0 - lmd1  - lmd2
        select pk
            
        case 1 then
            Phi = zeros(3,1);
            Phi(1) = lmd1 ; 
            Phi(2) = lmd2 ;
            Phi(3) = lmd3 ;
            
        case 2 then
            Phi = zeros(6,1);
            Phi(1) = lmd1.*( 2.0*lmd1 - 1.0) ; 
            Phi(2) = lmd2.*( 2.0*lmd2 - 1.0) ;
            Phi(3) = lmd3.*( 2.0*lmd3 - 1.0) ;    
            Phi(4) = 4.0*lmd1.*lmd2 ;
            Phi(5) = 4.0*lmd2.*lmd3 ;
            Phi(6) = 4.0*lmd3.*lmd1 ;
  
        case 3 then
            Phi = zeros(10,1);
            Phi(1)= lmd1.*(3.0*lmd1-2)*(3.0*lmd1-1)/2.0; 
            Phi(2)= lmd2.*(3.0*lmd2-2)*(3.0*lmd2-1)/2.0;
            Phi(3)= lmd3.*(3.0*lmd3-2)*(3.0*lmd3-1)/2.0;
            Phi(4)= 9.0*lmd1.*lmd2.*(3.0*lmd1-1)/2.0;
            Phi(5)= 9.0*lmd1.*lmd2.*(3.0*lmd2-1)/2.0;
        	Phi(6)= 9.0*lmd2.*lmd3.*(3.0*lmd2-1)/2.0;
            Phi(7)= 9.0*lmd2.*lmd3.*(3.0*lmd3-1)/2.0;
            Phi(8)= 9.0*lmd1.*lmd3.*(3.0*lmd3-1)/2.0;
            Phi(9)= 9.0*lmd1.*lmd3.*(3.0*lmd1-1)/2.0;
            Phi(10)= 27.0*lmd1.*lmd2.*lmd3;

          
      else
            printf("please choose either P1,P2 or P3"); 

    end
endfunction
    
//
// Gradients des fonctions de base Pk-Lagrange
// pour k= 1, 2 et 3
// --------------------------------------------
function GradPhi = GradPhi_Pk(lmd1, lmd2, G1, G2, pk)
    lmd3 = 1.0 - lmd1  - lmd2;
    G3   = - G1 - G2;
    
    select pk
        
    case 1 then
        GradPhi(1,:) = G1 ; 
        GradPhi(2,:) = G2 ;
        GradPhi(3,:) = G3 ;
        
    case 2 then
        GradPhi = zeros(6,2)

        GradPhi(1,:) = G1*(2.0*lmd1 - 1.0) + lmd1*(2.0*G1)
        GradPhi(2,:) = G2*(2.0*lmd2 - 1.0) + lmd2*(2.0*G2)
        GradPhi(3,:) = G3*(2.0*lmd3 - 1.0) + lmd3*(2.0*G3)

        GradPhi(4,:) = 4.0*(G1*lmd2 + lmd1*G2)
        GradPhi(5,:) = 4.0*(G2*lmd3 + lmd2*G3)
        GradPhi(6,:) = 4.0*(G3*lmd1 + lmd3*G1)


    case 3 then
        GradPhi = zeros(10,2)

        GradPhi(1,:) = 0.5*(G1*(3.0*lmd1 - 2.0)*(3.0*lmd1 -1.0) + lmd1*(3.0*G1)*(3.0*lmd1 -1.0)+ lmd1*(3.0*lmd1 - 2.0)*(3.0*G1))
        GradPhi(2,:) = 0.5*(G2*(3.0*lmd2 - 2.0)*(3.0*lmd2 -1.0) + lmd2*(3.0*G2)*(3.0*lmd2 -1.0)+ lmd2*(3.0*lmd2 - 2.0)*(3.0*G2))
        GradPhi(3,:) = 0.5*(G3*(3.0*lmd3 - 2.0)*(3.0*lmd3 -1.0) + lmd3*(3.0*G3)*(3.0*lmd3 -1.0)+ lmd3*(3.0*lmd3 - 2.0)*(3.0*G3))

        GradPhi(4,:) = (9.0/2.0)*(G1*lmd2*(3.0*lmd1-1)+lmd1*G2*(3.0*lmd1-1)+lmd1*lmd2*(3.0*G1))
        GradPhi(5,:) = (9.0/2.0)*(G1*lmd2*(3.0*lmd2-1)+lmd1*G2*(3.0*lmd2-1)+lmd1*lmd2*(3.0*G2))
        GradPhi(6,:) = (9.0/2.0)*(G2*lmd3*(3.0*lmd2-1)+lmd2*G3*(3.0*lmd2-1)+lmd2*lmd3*(3.0*G2))
        GradPhi(7,:) = (9.0/2.0)*(G2*lmd3*(3.0*lmd3-1)+lmd2*G3*(3.0*lmd3-1)+lmd2*lmd3*(3.0*G3))
        GradPhi(8,:) = (9.0/2.0)*(G1*lmd3*(3.0*lmd3-1)+lmd1*G3*(3.0*lmd3-1)+lmd1*lmd3*(3.0*G3))
        GradPhi(9,:) = (9.0/2.0)*(G1*lmd3*(3.0*lmd1-1)+lmd1*G3*(3.0*lmd1-1)+lmd1*lmd3*(3.0*G1))

        GradPhi(10,:) = 27.0*(G1*lmd2*lmd3+lmd1*G2*lmd3+lmd1*lmd2*G3)

        
    else
        printf("please choose either P1,P2 or P3");
    end
endfunction
//-----------------------------------------
// déterminant de deux vecteurs 2D
//-----------------------------------------
function out = Determinant(p,q)
    out = p(1)*q(2) - p(2)*q(1)
endfunction

//----------------------------------------- 
// Coordonnées barycentriques 
// au point Xc  dans le triangle ( Xi, Xj, Xk )
// -----------------------------------------------------------------
function Lambda=Lambda_of_P1(Xc, Xi, Xj, Xk)
    Lambda(1) = Determinant(Xc-Xj,Xk-Xj)/Determinant(Xi-Xj,Xk-Xj);
    Lambda(2) = Determinant(Xc-Xk,Xi-Xk)/Determinant(Xj-Xk,Xi-Xk);
    Lambda(3) = Determinant(Xc-Xi,Xj-Xi)/Determinant(Xk-Xi,Xj-Xi);
endfunction
//------------------------------------------------------------------
// Gradient des Coordonnées barycentriques
//--------------------------------------------------------------
function [Grad]=MGrad_Lambda_of_P1(Xi, Xj, Xk)
    
     e1=[1,0] ; e2 =[0,1]
    
    Grad(1,1) = Determinant(e1,Xk-Xj)/Determinant(Xi-Xj,Xk-Xj);
    Grad(1,2) = Determinant(e2,Xk-Xj)/Determinant(Xi-Xj,Xk-Xj);  

    Grad(2,1) = Determinant(e1,Xi-Xk)/Determinant(Xj-Xk,Xi-Xk);
    Grad(2,2) = Determinant(e2,Xi-Xk)/Determinant(Xj-Xk,Xi-Xk);

    Grad(3,1) = Determinant(e1,Xj-Xi)/Determinant(Xk-Xi,Xj-Xi);
    Grad(3,2) = Determinant(e2,Xj-Xi)/Determinant(Xk-Xi,Xj-Xi);
  
endfunction
//---------------------------------------------------------------
//--------------------------------
//--------------------------------
//---------------------------------------------------------------
//  donnees pour integration avec deux points de Gauss
// -------------------------------------------------
function [Poid, Xsi, Ngo]= IntegrationNum(Ngi)
    Ngo  = Ngi

    select Ngi
        
    case 1 then// integration numérique avec UN points de Gauss.
       
        Poid = zeros(Ngo,1);        // column vector
        Xsi  = zeros(3, Ngo);       // 3 x Ngo matrix

        Poid(1) = 0.5;

        Xsi(1,:) = 1.0/3.0;
        Xsi(2,:) = 1.0/3.0;

        
    case 2 then// integration numérique avec DEUX points de Gauss.
        Poid = zeros(Ngo,1);
        Xsi  = zeros(3, Ngo);

        // weights
        Poid(1) = 0.25;
        Poid(2) = 0.25;

        // Gauss points (barycentric coordinates)
        Xsi(:,1) = [0.5; 0.5; 0.0];
        Xsi(:,2) = [0.5; 0.0; 0.5];

    case 3 then// integration numérique avec TROIS points de Gauss.
        Poid = zeros(Ngo,1);
        Xsi  = zeros(3, Ngo);
        
        // weights
        Poid(1) = 1.0/6.0;
        Poid(2) = 1.0/6.0;
        Poid(3) = 1.0/6.0;
        
        // Gauss points (barycentric coordinates)
        Xsi(:,1) = [1.0/6.0; 1.0/6.0; 2.0/3.0];
        Xsi(:,2) = [2.0/3.0; 1.0/6.0; 1.0/6.0];
        Xsi(:,3) = [1.0/6.0; 2.0/3.0; 1.0/6.0];

    case 4 then// integration numérique avec TROIS points de Gauss.
        Poid = zeros(Ngo,1);
        Xsi  = zeros(3, Ngo);
        
        // weights
        Poid(1) = -27.0/96.0;
        Poid(2) =  25.0/96.0;
        Poid(3) =  25.0/96.0;
        Poid(4) =  25.0/96.0;
        
        // Gauss points (barycentric coordinates)
        // barycenter
        Xsi(:,1) = [1.0/3.0; 1.0/3.0; 1.0/3.0];
        
        // symmetric points
        Xsi(:,2) = [0.6; 0.2; 0.2];
        Xsi(:,3) = [0.2; 0.6; 0.2];
        Xsi(:,4) = [0.2; 0.2; 0.6];

    else
        // une autre alternative
        // six points de Gauss
        // -------------------------------------------------------
        // Nombre de points de Gauss
        // --------------------------
        Ngo               = 6;

        s1 = 0.11169079483905;
        s2 = 0.0549758718227661;
        aa = 0.445948490915965;
        bb = 0.091576213509771;
        // poids de Gauss 
        // -------------------------------------
        Poid(1:3)             = s2; 
        Poid(4:6)             = s1;

        // Points de Gauss xi_1 == Xsi_1
        // -------------------------------------
        Xsi(1,1:Ngo)         = [bb, 1-2*bb, bb, aa, aa, 1-2*aa];
        // Points de Gauss xi_2 == Xsi_2
        // -------------------------------------
        Xsi(2,1:Ngo)         = [bb, bb, 1-2*bb, 1-2*aa, aa, aa];
    end 
    
     
    //   Xsi_3 = 1 - Xsi_1 -Xsi_2
    // =======================================
    Xsi(3, 1:Ngo)  = 1.0 - Xsi(1,1:Ngo) - Xsi(2,1:Ngo);

endfunction
//------------------------------------------------------------

//---------------------------------------------------------------
//---------------------------------------------------------------
// points de Gauss et poids de Gauss 
//   dans l'élément de référence
//---------------------------------------------------------------
//---------------------------------------------------------------
//------------------------------------------------------------
// -------------------------------------------------------
// Nombre de points de Gauss
// --------------------------
Ngi                 = 60;
//
//=========================================================
//   Choix de la méthode Eléments finis : 2D, Pk-Lagrange
//--------------------------------------------------------
//   Pk  avec k = EF_Pk 
// ------------------------
//   EF_Pk = 1 <===> P1-Lagrange
//   EF_Pk = 2 <===> P2-Lagrange
//   EF_Pk = 3 <===> P3-Lagrange
//=========================================================
// ============================================
// Interactive start — stop execution first
// ============================================
if ~exists("RUN_SIMULATION","local") then
    mprintf("\n")    
    disp("=======================================");
    disp(" FEM 2D solver");
    disp(" Choose parameters, then run again");
    disp("=======================================\n");

    mprintf("Choose pb:\n");
    mprintf("  1: reaction (alpha=1)\n");
    mprintf("  2: diffusion iso (4,4)\n");
    mprintf("  3: diffusion anis (1,2)\n");
    mprintf("  4: diffusion anis (1,0)\n");
    mprintf("  5: diffusion anis (1e-8,1)\n");
    mprintf("  6: convection (0.5,1) + optional stabilization dh^m\n");
    mprintf("  7: conv-reac-diff anis\n\n");
    pb = input("pb = ");

    mprintf("\nChoose FE order:\n");
    mprintf("  1: P1\n  2: P2\n  3: P3\n\n");
    EF_Pk = input("EF_Pk = ");

    i    = 0;
    mExp = 1;

    if pb == 6 then
        mprintf("\nStabilization for convection problem:\n");
        mprintf("  0 : No stabilization (Diff = 0)\n");
        mprintf("  1 : Yes (Diff = diag(dh^m, dh^m))\n\n");
    
        i = input("Choose i (0 or 1): ");
    
        if i == 1 then
            mExp = input("Enter exponent m in dh^m (e.g. 1, 2, 3): ");
        else
            mExp = 1;   // default value, not used
        end
    end

    // Store choices
    RUN_SIMULATION = %t;

    mprintf("\nChosen parameters: pb=%d, EF_Pk=%d, i=%d, mExp=%g\n", pb, EF_Pk, i, mExp);
    mprintf("Now run the script again to start computation.\n\n");
    return;
end
Lx  =1   ; Ly=1 
Nx0 =10  ; Ny0=10

kx = %pi/Lx
ky = 2*%pi/Ly
//ky = 0.00*%pi/Ly

[W, Lambda, Ngp]= IntegrationNum(Ngi)

// Analyse de convergence
// -----------------------------
Nconv = 5
Vconv = [13,25, 31, 61, 73, 91; ...
7,13, 16, 31, 37, 46; ...
5, 9, 11, 21, 25, 31  ]
//[5,11,21,31,9,21,31,61,17,31,61,121]
for rf = 1:Nconv

    sf = rf
    Nx = Vconv(EF_Pk,sf)
    Ny = Nx
    //---------------------------------------------------------------
    // Construction du maillage Pk pour pk = EF_Pk
    //---------------------------------------------------------------
    [Pk_Np, Pk_Ne, Pk_Coor, Pk_Nu, Pk_LogP, Pk_LogE, Pk_NuV] ...
    = Struct_Pk_Mesh(Nx, Ny, Lx, Ly,EF_Pk)
    Pk_Npl = 3 ; 
    Pk_Ndl = (EF_Pk+1)*(EF_Pk +2)/2.0

    // nombre  global de degré de liberté 
    Nddl = Pk_Np   ;
    Coor = Pk_Coor ;
    LogP = Pk_LogP ;

    // nombre local (sur chaque éléments) de degré de liberté 
    Npl = Pk_Npl;
    Ndl = Pk_Ndl;
    Ne  = Pk_Ne ;
    Nu  = Pk_Nu ;

    // Initialisation de la matrice globale
    MatA(1:Nddl,1:Nddl) = 0.0;

    // Initialisation du second membre gobal
    VecB(1:Nddl)        = 0.0;

    //=========================================================
    // Assemblage 
    //=========================================================

    // boucle sur les éléments
    // ----------------------------------------------------
    for e =1:Ne

        Xi = Coor(:, Nu(1,e));
        Xj = Coor(:, Nu(2,e));
        Xk = Coor(:, Nu(3,e));

        DetE = abs(Determinant(Xj-Xi,Xk-Xi)); 

        // Boucle sur les points de Gauss
        // ----------------------------------------------------------
        for gp = 1:Ngp

            // Point physique associé au point de Gauss
            // ------------------------------------------
            Xg  = Lambda(1,gp)*Xi + Lambda(2,gp)*Xj ...
            + Lambda(3,gp)*Xk ;
            wg  = W(gp);

            [Coef_Lm, Coef_Reac, Coef_Mu] = Composite_Mat(Xg,pb,Lx,Ly, Ne,i,mExp); //pour le test de convergence pour le pb de convection, il faut ajouter les arg i et mExp 
            bx = Coef_Mu(1,1);
            by = Coef_Mu(2,2);
            cx = Coef_Lm(1);
            cy = Coef_Lm(2);
            alpha = Coef_Reac;


            GradP1         =  MGrad_Lambda_of_P1(Xi, Xj, Xk);


            // Fonctions de base Pk
            // *****************************************
            Phi      = Phi_Pk(Lambda(1,gp), Lambda(2,gp), EF_Pk)

            // Gradient des fonctions de base Pk
            // *********************************************
            GradPhi  = GradPhi_Pk(Lambda(1,gp), Lambda(2,gp), ...
            GradP1(1,:),  GradP1(2,:),EF_Pk)

            // -----------------------------------------------------
            // Second membre local,  Matrice Locale
            //           et assemblage
            // -----------------------
            // Boucle sur la numérotation locale (lignes)
            for k=1:Ndl
                // numérotation globale (is) de l'indice local (k)
                is          = Nu(k,e)       ;
                Phi_is      = Phi(k)        ; 
                GradPhi_is  = GradPhi(k, :) ;
                // -------------------------------------------------
                //      second membre local (k)
                //      -------------------------
                // !!!! A COMPLETER si besoin : le calcul de Be_k  
                // -------------------------------------------------
                Be_k        = f(Xg,kx,ky,bx,by,cx,cy,alpha)*Phi_is ;
                
                // -----------------------------------
                // Assemblage du second membre global (is)
                // ------------------------------------
                VecB(is)    = VecB(is) + wg*DetE*Be_k ;

                // Boucle sur la numérotation locale (colonnes)
                for kp = 1:Ndl
                    // numérotation globale (js) de l'indice local (kp)
                    js         = Nu(kp,e)      ;
                    Phi_js     = Phi(kp)       ; 
                    GradPhi_js = GradPhi(kp,:) ;
                    // -------------------------------------------------
                    //      matrice  locale (k, kp)
                    //      -------------------------
                    // A COMPLETER si besoin: le calcul de Ae_k_kp  !!
                    // -------------------------------------------------
                    Ae_k_kp    = ( GradPhi_is* Coef_Mu * GradPhi_js')                                                                                                   + Coef_Reac * Phi_is * Phi_js                                                                                                       - Phi_js * (Coef_Lm' * GradPhi_is');
                            
                    // ---------------------------------
                    // Assemblage de la matrice globale (is, js)
                    // ----------------------------------
                    MatA(is,js)= MatA(is,js) + wg*DetE*Ae_k_kp ;
                end // sur (kp)
            end  // sur (k) 
        end  // sur (g) 
    end  // sur (e)      
    // -------------------------------------------------
    // A COMPLETER si besoin: la prise en compte des   !!
    //                         conditions aux limites  !!
    // -------------------------------------------------
    // Conditions aux limites (peut être optimisé)
    //    Dirichlet homogène sur le bord gauche
    //    Neumann sur tous les autres bords
    // -------------------------------------------
    for is=1:Nddl
        Xg = Coor(:, is);

        if LogP(is) == -1 then
            MatA(is, :)    = 0.0
            MatA(is, is)   = 1.0
            VecB(is)       = Exact(Xg(1), Xg(2), kx, ky)
        end 
        if LogP(is) == -2 then
            MatA(is, :)  = 0.0
            MatA(is, is) = 1.0
            VecB(is)     = Exact(Xg(1), Xg(2), kx, ky)
        end 
        if LogP(is) < 0 then
            MatA(is, :)    = 0.0
            MatA(is, is)   = 1.0
            VecB(is)       = Exact(Xg(1), Xg(2), kx, ky)
        end     
    end


    // Version de résolution du système en 
    // utilisant une matrice creuse et umfpack
    // ---------------------------------------
    P1_Sparse_Mat = sparse(MatA); 
    VecSol        = umfpack(P1_Sparse_Mat,"\",VecB)
    //VecSol = MatA\VecB;

    //
    // Visualisation
    // -------------------------------------------
    Pk_x=Coor(1,:)';
    Pk_y=Coor(2,:)';


    je = 0
    for ie=1:Ne
        for jl = 1:int(size(Pk_NuV,2) )
            i1 = Pk_NuV(1,jl) ; 
            i2 = Pk_NuV(2,jl) ; 
            i3 = Pk_NuV(3,jl)  ;
            je = je +1
            Pk_P1_T(1:5,je) =[je,Nu(i1,ie),Nu(i2,ie), Nu(i3,ie),je] 
        end
    end


    Pk_exact    = Exact(Pk_x,Pk_y,kx,ky);
    scf(rf);        // figure # = refinement index
    clf(rf);        // clear that figure
    gcf().color_map = jetcolormap(64);
    fec(Pk_x, Pk_y, Pk_P1_T', VecSol, mesh=%t);
    xtitle(msprintf("rf=%d, Nx=%d, Ne=%d", rf, Nx, Ne));

    // boucle sur les éléments
    // ----------------------------------------------------
    // Calcul de l'erreur
    // ------------------------
    h   = Lx*Ly/Ne;
    H(rf)= sqrt(Lx*Ly/Ne)
    NddLC(rf)= Nddl; 
    VNx(rf)= Nx;
    ErrL1(rf)   = 0;   ErrL2(rf)   = 0; ErrLinf(rf) = 0.0 ;
    // boucle sur les éléments
    for e = 1:Ne
        // Boucle sur les points de Gauss
        Xi = Coor(:, Nu(1,e));
        Xj = Coor(:, Nu(2,e));
        Xk = Coor(:, Nu(3,e));

        DetE          = abs(Determinant(Xi-Xj,Xk-Xj))      ;
        // Boucle sur les points de Gauss
        // ----------------------------------------------------------
        for gp = 1:Ngp
            // Fonctions de base Pk
            // *****************************************
            Phi      = Phi_Pk(Lambda(1,gp), Lambda(2,gp), EF_Pk)
            wg   = W(gp);

            // Point physique associé au point de Gauss
            // ------------------------------------------        
            Xg(1:2) = 0 ;
            for k=1:Npl
                is      = Nu(k,e)     ;     
                Xg      = Xg + Coor(:, is)*Lambda(k,gp); 
            end 

            // Solution approchée au point de Gauss
            // ------------------------------------------        
            Ug = 0
            for k=1:Ndl
                is   = Nu(k,e)                 ;       
                Ug   = Ug +   VecSol(is)*Phi(k);
            end 

            // Solution exacte au point de Gauss
            // -----------------------------------------
            Uex  = Exact(Xg(1), Xg(2), kx, ky )    ;

            // Les Erreurs L1, L2 et Linf
            // ----------------------------------------------
            ErrL1(rf)   = ErrL1(rf) + wg*DetE*abs(Ug -Uex)  ; 
            ErrL2(rf)   = ErrL2(rf) + wg*DetE*(Ug -Uex)^2  ;
            ErrLinf(rf) = max( ErrLinf(rf), abs(Ug -Uex) ) ; 
        end     
    end
    ErrL2(rf) = sqrt(ErrL2(rf))

end

Dh                 = H(1:Nconv-1)./H(2:Nconv)
Ordre_de_Conv(1,:) = log(ErrL1(1:Nconv-1)./ErrL1(2:Nconv))./log(Dh)
Ordre_de_Conv(2,:) = log(ErrL2(1:Nconv-1)./ErrL2(2:Nconv))./log(Dh)
Ordre_de_Conv(3,:) = log(ErrLinf(1:Nconv-1)./ErrLinf(2:Nconv))./log(Dh)
disp(EF_Pk)
disp([... //"Pk" , 
"Nddl",  "Norme L1","Norme L2","Norme Linf", "Erreur"] )
disp([... //EF_Pk*ones(1,Nconv-1); 
NddLC(2:Nconv)';...//VNx(2:Nconv)';
Ordre_de_Conv;ErrL2(2:Nconv)']')

