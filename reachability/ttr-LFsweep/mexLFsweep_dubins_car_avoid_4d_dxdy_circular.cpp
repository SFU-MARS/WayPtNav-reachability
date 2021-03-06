#include <math.h>
#include "matrix.h"
#include "mex.h"   //--This one is required
#define min(a,b) (((a) < (b)) ? (a) : (b))
#define max(a,b) (((a) > (b)) ? (a) : (b))

double sweepingLF(double**** phi, int N1, int N2, int N3, int N4, double***** xs, double* dx, double a_max, double omega_max, double d_max_xy, double d_max_theta)
{
    int i,j,k,l,s1,s2,s3,s4;
    double c, p, q, r, m, H, phiTemp, phiOld, error;
    double a, omega, d_x, d_y, d_theta; // optimal control (a, omega) and worst disturbance (d_xy, d_theta)
    double sigma1, sigma2, sigma3, sigma4;
//     double small=0.1;
//     double pP, pM, qP, qM, rP, rM, HP, HM, sigmaP, sigmaM;
    
    error = 0;
//     N1 = N[0]; N2 = N[1]; N3 = N[2];
    
    for (s1=1; s1>=-1; s1-=2 )
    for (s2=1; s2>=-1; s2-=2 )
    for (s3=1; s3>=-1; s3-=2 )
    for (s4=1; s4>=-1; s4-=2 )
    {
        // LF sweeping module
        for ( i=(s1<0 ? (N1-2):1); (s1<0 ? i>=1:i<=(N1-2)); i+=s1 )
        for ( j=(s2<0 ? (N2-2):1); (s2<0 ? j>=1:j<=(N2-2)); j+=s2 ) 
        for ( k=(s3<0 ? (N3-1):0); (s3<0 ? k>=0:k<=(N3-1)); k+=s3 )
        for ( l=(s4<0 ? (N4-2):1); (s4<0 ? l>=1:l<=(N4-2)); l+=s4 ) 
        {

            phiOld = phi[i][j][k][l];

            if (k == 0) // For angle dimension, consider the boudary extension to another side
            {
                
                p = (phi[i+1][j][k][l] - phi[i-1][j][k][l])/(2*dx[0]); // phi(i+1) - phi(i-1)
                q = (phi[i][j+1][k][l] - phi[i][j-1][k][l])/(2*dx[1]); // phi(j+1) - phi(j-1)
                r = (phi[i][j][k+1][l] - phi[i][j][N3-1][l])/(2*dx[2]); // phi(k+1) - phi(k-1)
                m = (phi[i][j][k][l+1] - phi[i][j][k][l-1])/(2*dx[3]);
                
                // Control: maximize, Disturbance: minimize
                if (m >= 0) {
                    a = a_max;
                } else { 
                    a = - a_max;
                } 
                if (r >= 0) {
                    omega = omega_max;
                } else { 
                    omega = - omega_max;
                }  
                // Compute worst disturbance dx, dy, circular constraints
                if (abs(p) > abs(q)) {
                    d_x = (p > 0 ? (-d_max_xy):d_max_xy);
                    d_y = 0;
                } else {
                    d_y = (q > 0 ? (-d_max_xy):d_max_xy);
                    d_x = 0;
                }
                if (r >= 0) {
                    d_theta = - d_max_theta;
                } else {
                    d_theta = d_max_theta;
                }

                sigma1 = abs(xs[i][j][k][l][3]*cos(xs[i][j][k][l][2])) + abs(d_x); // upper bound of 
                sigma2 = abs(xs[i][j][k][l][3]*sin(xs[i][j][k][l][2])) + abs(d_y);
                sigma3 = abs(omega) + d_max_theta;
                sigma4 = abs(a);

                c = (sigma1/dx[0] + sigma2/dx[1] + sigma3/dx[2] + sigma4/dx[3]);

                H = (-1) * ((xs[i][j][k][l][3]*cos(xs[i][j][k][l][2]) + d_x)*p + (xs[i][j][k][l][3]*sin(xs[i][j][k][l][2]) + d_y)*q + (omega + d_max_theta)*r + a*m + 1);
                
                phiTemp = - H + sigma1*(phi[i+1][j][k][l] + phi[i-1][j][k][l])/(2*dx[0])\
                        + sigma2*(phi[i][j+1][k][l] + phi[i][j-1][k][l])/(2*dx[1])\
                        + sigma3*(phi[i][j][k+1][l] + phi[i][j][N3-1][l])/(2*dx[2])\
                        + sigma4*(phi[i][j][k][l+1] + phi[i][j][k][l-1])/(2*dx[3]);
            }
            else if (k == N3-1)
            {
                p = (phi[i+1][j][k][l] - phi[i-1][j][k][l])/(2*dx[0]); // phi(i+1) - phi(i-1)
                q = (phi[i][j+1][k][l] - phi[i][j-1][k][l])/(2*dx[1]); // phi(j+1) - phi(j-1)
                r = (phi[i][j][0][l] - phi[i][j][k-1][l])/(2*dx[2]); // phi(k+1) - phi(k-1)
                m = (phi[i][j][k][l+1] - phi[i][j][k][l-1])/(2*dx[3]);
                
                // Control: maximize, Disturbance: minimize
                if (m >= 0) {
                    a = a_max;
                } else { 
                    a = - a_max;
                } 
                if (r >= 0) {
                    omega = omega_max;
                } else { 
                    omega = - omega_max;
                }  
                // Compute worst disturbance dx, dy, circular constraints
                if (abs(p) > abs(q)) {
                    d_x = (p > 0 ? (-d_max_xy):d_max_xy);
                    d_y = 0;
                } else {
                    d_y = (q > 0 ? (-d_max_xy):d_max_xy);
                    d_x = 0;
                }
                if (r >= 0) {
                    d_theta = - d_max_theta;
                } else {
                    d_theta = d_max_theta;
                }

                sigma1 = abs(xs[i][j][k][l][3]*cos(xs[i][j][k][l][2])) + abs(d_x); // upper bound of 
                sigma2 = abs(xs[i][j][k][l][3]*sin(xs[i][j][k][l][2])) + abs(d_y);
                sigma3 = abs(omega) + d_max_theta;
                sigma4 = abs(a);

                c = (sigma1/dx[0] + sigma2/dx[1] + sigma3/dx[2] + sigma4/dx[3]);

                H = (-1) * ((xs[i][j][k][l][3]*cos(xs[i][j][k][l][2]) + d_x)*p + (xs[i][j][k][l][3]*sin(xs[i][j][k][l][2]) + d_y)*q + (omega + d_max_theta)*r + a*m + 1);
                
                phiTemp = - H + sigma1*(phi[i+1][j][k][l] + phi[i-1][j][k][l])/(2*dx[0]) +\
                        sigma2*(phi[i][j+1][k][l] + phi[i][j-1][k][l])/(2*dx[1]) +\
                        sigma3*(phi[i][j][0][l] + phi[i][j][k-1][l])/(2*dx[2]) +\
                        sigma4*(phi[i][j][k][l+1] + phi[i][j][k][l-1])/(2*dx[3]);
            }
            else
            {
                p = (phi[i+1][j][k][l] - phi[i-1][j][k][l])/(2*dx[0]); // phi(i+1) - phi(i-1)
                q = (phi[i][j+1][k][l] - phi[i][j-1][k][l])/(2*dx[1]); // phi(j+1) - phi(j-1)
                r = (phi[i][j][k+1][l] - phi[i][j][k-1][l])/(2*dx[2]); // phi(k+1) - phi(k-1)
                m = (phi[i][j][k][l+1] - phi[i][j][k][l-1])/(2*dx[3]);
                
                // Control: maximize, Disturbance: minimize
                if (m >= 0) {
                    a = a_max;
                } else { 
                    a = - a_max;
                } 
                if (r >= 0) {
                    omega = omega_max;
                } else { 
                    omega = - omega_max;
                }  
                // Compute worst disturbance dx, dy, circular constraints
                if (abs(p) > abs(q)) {
                    d_x = (p > 0 ? (-d_max_xy):d_max_xy);
                    d_y = 0;
                } else {
                    d_y = (q > 0 ? (-d_max_xy):d_max_xy);
                    d_x = 0;
                }
                if (r >= 0) {
                    d_theta = - d_max_theta;
                } else {
                    d_theta = d_max_theta;
                }

                sigma1 = abs(xs[i][j][k][l][3]*cos(xs[i][j][k][l][2])) + abs(d_x); // upper bound of 
                sigma2 = abs(xs[i][j][k][l][3]*sin(xs[i][j][k][l][2])) + abs(d_y);
                sigma3 = abs(omega) + d_max_theta;
                sigma4 = abs(a);

                c = (sigma1/dx[0] + sigma2/dx[1] + sigma3/dx[2] + sigma4/dx[3]);

                H = (-1) * ((xs[i][j][k][l][3]*cos(xs[i][j][k][l][2]) + d_x)*p + (xs[i][j][k][l][3]*sin(xs[i][j][k][l][2]) + d_y)*q + (omega + d_max_theta)*r + a*m + 1);
                
                phiTemp = - H + sigma1*(phi[i+1][j][k][l] + phi[i-1][j][k][l])/(2*dx[0]) +\
                        sigma2*(phi[i][j+1][k][l] + phi[i][j-1][k][l])/(2*dx[1]) +\
                        sigma3*(phi[i][j][k+1][l] + phi[i][j][k-1][l])/(2*dx[2]) +\
                        sigma4*(phi[i][j][k][l+1] + phi[i][j][k][l-1])/(2*dx[3]);
            }
            
            phi[i][j][k][l] = min(phiTemp/c, phiOld);

            error = max(error, phiOld - phi[i][j][k][l]);

        }
        // computational boundary condition
        
        for ( j = 0; j <= (N2-1); j++)
        for ( k = 0; k <= (N3-1); k++)
        for ( l = 0; l <= (N4-1); l++)
        {
            phiOld = phi[0][j][k][l]; 
            phi[0][j][k][l] = min(max(2*phi[1][j][k][l] - phi[2][j][k][l], phi[2][j][k][l]), phiOld); 
            error = max(error, phiOld - phi[0][j][k][l]);
            
            phiOld = phi[N1-1][j][k][l];
            phi[N1-1][j][k][l] = min(max(2*phi[N1-2][j][k][l] - phi[N1-3][j][k][l], phi[N1-3][j][k][l]), phiOld); 
            error = max(error, phiOld - phi[N1-1][j][k][l]);
        }

        for ( k = 0; k <= (N3-1); k++)
        for ( i = 0; i <= (N1-1); i++)
        for ( l = 0; l <= (N4-1); l++)
        {
            phiOld = phi[i][0][k][l]; 
            phi[i][0][k][l] = min(max(2*phi[i][1][k][l] - phi[i][2][k][l], phi[i][2][k][l]), phiOld); 
            error = max(error, phiOld - phi[i][0][k][l]);
            
            phiOld = phi[i][N2-1][k][l];
            phi[i][N2-1][k][l] = min(max(2*phi[i][N2-2][k][l] - phi[i][N2-3][k][l], phi[i][N2-3][k][l]), phiOld); 
            error = max(error, phiOld - phi[i][N2-1][k][l]);
            
        }
        
        for ( k = 0; k <= (N3-1); k++)
        for ( i = 0; i <= (N1-1); i++)
        for ( j = 0; j <= (N2-1); j++)
        {
            phiOld = phi[i][j][k][0]; 
            phi[i][j][k][0] = min(max(2*phi[i][j][k][1] - phi[i][j][k][2], phi[i][j][k][2]), phiOld); 
            error = max(error, phiOld - phi[i][j][k][0]);
            
            phiOld = phi[i][j][k][N4-1];
            phi[i][j][k][N4-1] = min(max(2*phi[i][j][k][N4-2] - phi[i][j][k][N4-3], phi[i][j][k][N4-3]), phiOld); 
            error = max(error, phiOld - phi[i][j][k][N4-1]);
            
        }       
    }
    return error;
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //---Inside mexFunction---
    //Declarations
    double *phiValues, *xsValues, *dxValues;
    double ****phi, *****xs, *dx;
    const mwSize *N;
	int numIter;
    double a_max,omega_max,d_max_xy,d_max_theta,error,TOL;
    int i,j,k,l,m, N1,N2,N3,N4;

    //double L = 1;
    double numInfty = 1000;
    
    //Get the input
    phiValues    = mxGetPr(prhs[0]);
    xsValues    = mxGetPr(prhs[1]);
    dxValues    = mxGetPr(prhs[2]);
    a_max       = (double)mxGetScalar(prhs[3]);
    omega_max          = (double)mxGetScalar(prhs[4]);
    d_max_xy          = (double)mxGetScalar(prhs[5]);
    d_max_theta          = (double)mxGetScalar(prhs[6]);
    numIter     = (int)mxGetScalar(prhs[7]);
    TOL         = (double)mxGetScalar(prhs[8]);
    
    N           = mxGetDimensions(prhs[0]);
    
    N1 = N[0]; N2 = N[1]; N3 = N[2]; N4 = N[3];
    
    // memory allocation & value assignment
	phi   = (double ****) malloc ( N1 * sizeof(double***));
    dx    = (double *) malloc ( 4 * sizeof(double));
    xs    = (double *****) malloc ( N1 * sizeof(double****));
    
	for (i=0; i<N1; i++)
    {
		phi[i]   = (double ***) malloc ( N2 * sizeof(double**));
        xs[i]    = (double ****) malloc ( N2 * sizeof(double***));
        for (j=0; j<N2; j++)
        {
            phi[i][j]   = (double **) malloc ( N3 * sizeof(double*));
            xs[i][j]    = (double ***) malloc ( N3 * sizeof(double**));
            for (k=0; k<N3; k++)
            {
                phi[i][j][k]   = (double *) malloc ( N4 * sizeof(double));
                xs[i][j][k]    = (double **) malloc ( N4 * sizeof(double*));
                for (l=0; l<N4; l++)
                {
                    phi[i][j][k][l]   = phiValues[((l*N3+k)*N2+j)*N1+i];
                    xs[i][j][k][l]    = (double *) malloc ( 4 * sizeof(double));
                    for (m=0; m<4; m++) {
                        xs[i][j][k][l][m]      = xsValues[(((m*N4+l)*N3+k)*N2+j)*N1+i];
                    }
                }
            }
        }
    }

    for (i=0; i<4; i++)
    {
        dx[i] = dxValues[i];
    }
    
    // run LF sweeping algorithm
    for(k=0; k<numIter; k++) 
    {
        error = sweepingLF(phi, N1, N2, N3, N4, xs, dx, a_max, omega_max, d_max_xy, d_max_theta);
        
        mexPrintf("Error = %g at iteration %i. \n", error, k);
        if (error <= TOL) {
            mexPrintf("Stopped at iteration %i. \n", k);
            break;
        } 
        
    }
  
    // send the processed phi to the output  
    for (i=0; i < N1; i++)
	for (j=0; j < N2; j++)
    for (k=0; k < N3; k++)
    for (l=0; l < N4; l++)
        phiValues[((l*N3+k)*N2+j)*N1+i] = phi[i][j][k][l];
    
    // delete memory;
	for(i=0; i< N1; i++)
    {
        for(j=0; j<N2; j++)
        {
            for(k=0; k<N3; k++)
            {
                for(l=0; l<N4; l++) 
                {
                    free(xs[i][j][k][l]);
                }
                free(phi[i][j][k]);
                free(xs[i][j][k]);
            }
            free(phi[i][j]);
            free(xs[i][j]);
        }
        free(phi[i]);
        free(xs[i]);
	}
	free(phi);
    free(xs);
    free(dx);
}