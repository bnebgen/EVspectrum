/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-2013, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.

   SLEPc is free software: you can redistribute it and/or modify it under  the
   terms of version 3 of the GNU Lesser General Public License as published by
   the Free Software Foundation.

   SLEPc  is  distributed in the hope that it will be useful, but WITHOUT  ANY
   WARRANTY;  without even the implied warranty of MERCHANTABILITY or  FITNESS
   FOR  A  PARTICULAR PURPOSE. See the GNU Lesser General Public  License  for
   more details.

   You  should have received a copy of the GNU Lesser General  Public  License
   along with SLEPc. If not, see <http://www.gnu.org/licenses/>.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Generalized Hamiltonian Solver";

#include <slepceps.h>
#include <petscblaslapack.h>
#include <stdio.h>
#include <stddef.h>

/*
   User-defined routines
*/
PetscErrorCode MatrixMult(Mat A,Vec x,Vec y);
PetscErrorCode MatrixDiagonal(Mat A,Vec diag);
static void vecsum(const int n, const PetscInt *x, const PetscInt *y,PetscInt *ret);
static void vecprod(const int n, const PetscInt *x, const PetscInt *y,PetscInt *ret);
PetscScalar vecdot(const int n, const PetscScalar *x, const PetscInt *y);
PetscInt vecmarkers(const int n, const PetscInt *x, const PetscInt *y);
PetscScalar vecpowers(const int n, const PetscInt *vib, const PetscInt *order);
PetscScalar raiselower(const int n,const PetscInt *colbf,const PetscInt *opp);
int ReadInputInt(FILE *inputfile, PetscInt *variable,int mode);
int ReadHamInt(char *inpstr,char termchar,PetscInt strdmode, PetscInt *j,PetscInt loc,PetscInt *variable);
int ReadInputFloat(FILE *inputfile,PetscScalar *variable,PetscInt index);


  struct data {
    PetscInt       nelec,nvib,size,vsize,hlength;
    PetscInt       *vbasis,*hvibes,*hfreq,*helec,*hvalref;
    PetscInt       *markers,*diag;
    PetscScalar    *hvalues,*gsfreq;
  };

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            B;               /* My matrix */
  Vec            testvec,testvec2,outfullvec;
  VecScatter     collectctx;
  EPS            eps;             /* eigenproblem solver context */
  EPSType        type;
  PetscMPIInt    size,rank;
  const PetscInt inpstrl=200;
  PetscInt       strdmode;
  PetscInt       N,n=10,nev=3,i=0,j=0;
  PetscInt       vlength=0,nconv;
  PetscInt       *hhermchk,*colbf;
  PetscInt       tempint,complete,actvib,curind;
  PetscScalar    *diagscal,*outscal;
  PetscScalar    leval,eval,tempscal,halfw,cutoff=.01,range=1000;
  PetscErrorCode ierr;
  FILE           *inputfile;
  FILE           *curoutfile;
  char           inpstr[inpstrl];
  char           tmpstr[10];
  char           curchar;

  struct data matdata;
  
  matdata.nvib=0;
  matdata.hlength=0;
  
  /*-------------------------------------------------------------------
  Initialize SLEPc
  ---------------------------------------------------------------------*/
  
  SlepcInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
//  if (size != 1) SETERRQ(PETSC_COMM_WORLD,1,"This is a uniprocessor example only");

  ierr = PetscOptionsGetInt(NULL,"-n",&n,NULL);CHKERRQ(ierr);
  N = n*n;
  
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Open and read in the input file
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
  
  
  inputfile=fopen("input.inp","r");
  if (!inputfile) {
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Failed to open file\n");CHKERRQ(ierr);
  }
  
  if(fgets(inpstr,inpstrl,inputfile)==NULL){
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nFirst Line\n");CHKERRQ(ierr);
  }
  
  
  while(strcmp(inpstr,"$problem\n")!=0){
    if(fgets(inpstr,inpstrl,inputfile)==NULL){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nBefore $problem section.\n");CHKERRQ(ierr);
	  return 101;
    }
  }
  
  
  while(strcmp(inpstr,"$end")!=0){
    if(fscanf(inputfile,"%s",inpstr)==0){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nIn $problem section.\n");CHKERRQ(ierr);
	  return 101;
    } 
    if (strcmp(inpstr,"nvib")==0) {
	  if(ReadInputInt(inputfile,&matdata.nvib,0)!=0){
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Error reading nvib.\n");CHKERRQ(ierr);
	    return 101;
      }
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Value of matdata.nvib is: %d \n",matdata.nvib);CHKERRQ(ierr);
	  PetscMalloc((matdata.nvib)*sizeof(PetscInt),&matdata.vbasis);
	} 
	if (strcmp(inpstr,"nelec")==0) {
	  if(ReadInputInt(inputfile,&matdata.nelec,0)!=0){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Error reading nelec.\n");CHKERRQ(ierr);
	    return 101;
      }
          ierr = PetscPrintf(PETSC_COMM_WORLD,"Value of matdata.nelec is: %d \n",matdata.nelec);
	} 
	if (strcmp(inpstr,"neigen")==0) {
	  if(ReadInputInt(inputfile,&nev,0)!=0){
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Error reading nev\n");CHKERRQ(ierr);
	    return 101;
      }
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Value of neigen is: %d \n",nev);CHKERRQ(ierr);
	}
	if(strcmp(inpstr,"vbasis")==0) {
	  if(matdata.nvib==0) {
		ierr = PetscPrintf(PETSC_COMM_WORLD,"Error reading matdata.vbasis\nmatdata.vbasis defined before matdata.nvib.\n");CHKERRQ(ierr);
		return 101;
	  }
	  i=0;
	  while(i<matdata.nvib) {
	   if(ReadInputInt(inputfile,&tempint,1)!=0){
		 ierr = PetscPrintf(PETSC_COMM_WORLD,"Error reading matdata.nvib.\n");CHKERRQ(ierr);
	     return 101;
       }
	   matdata.vbasis[i]=tempint;
	   i++;
	  }
    }
  }

  
  while(strcmp(inpstr,"$hamiltonian\n")!=0){
    if(fgets(inpstr,inpstrl,inputfile)==NULL){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nBefore $hamiltonian section.\n");CHKERRQ(ierr);
	  return 101;
    }
  }

  while(strcmp(inpstr,"$end\n")!=0){
    matdata.hlength++;
    if(fgets(inpstr,inpstrl,inputfile)==NULL){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nBefore end of $hamiltonian section.\n");CHKERRQ(ierr);
	  return 101;
    }
  }
  matdata.hlength--;
  
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of hamiltonian elements: %d \n",matdata.hlength);CHKERRQ(ierr);
  rewind(inputfile);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Start SLEPc
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  

  

  
  PetscMalloc(matdata.nvib*matdata.hlength*sizeof(PetscInt),&matdata.hvibes);
  PetscMalloc(matdata.nvib*matdata.hlength*sizeof(PetscInt),&matdata.hfreq);
  PetscMalloc(2*matdata.hlength*sizeof(PetscInt),&matdata.helec);
  PetscMalloc(matdata.hlength*sizeof(PetscInt),&matdata.hvalref);
  PetscMalloc(matdata.hlength*sizeof(PetscInt),&hhermchk);
  PetscMalloc(matdata.nvib*sizeof(PetscInt),&matdata.markers);
  PetscMalloc(matdata.hlength*sizeof(PetscInt),&matdata.diag);
  PetscMalloc(matdata.nvib*sizeof(PetscInt),&colbf);
  
  for(i=0;i<matdata.nvib*matdata.hlength;i++){
    matdata.hvibes[i]=0;
	matdata.hfreq[i]=0;
/*	printf("setting element %d to zero\n",i);*/
  }
  
  while(strcmp(inpstr,"$hamiltonian\n")!=0){
    if(fgets(inpstr,inpstrl,inputfile)==NULL){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nBefore $hamiltonian section");CHKERRQ(ierr);
	  return 101;
    }
  }
  
  strcpy(tmpstr,"          ");

  /*-----------------------------------------------
   Hamiltonian read loop
  -----------------------------------------------*/
  for(i=1;i<=matdata.hlength;i++){
    if(fgets(inpstr,inpstrl,inputfile)==NULL){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nBefore end of $hamiltonian section\n");CHKERRQ(ierr);
	  return 101;
    }
	j=0;
	curchar=inpstr[j];
	strdmode=0;
    while(j<inpstrl) {
	  if(curchar!='\n') {
		if(strdmode==0){
          if(curchar=='c'){
            strdmode=1;
          } else if (curchar=='|') {
            strdmode=2;
          } else if (curchar=='<') {
            strdmode=3;
          } else if (curchar=='n') {
            strdmode=7;
          } else if (curchar=='a') {
            j++;
            curchar=inpstr[j];
            if(curchar=='_'){
              strdmode=4;
            } else if(curchar=='^') {
              strdmode=5;
            } else {
			  ierr = PetscPrintf(PETSC_COMM_WORLD,"Error reading Hamiltonian.\nLine: %d Column: %d\nUnclear raising or lowering operator.\n",i,j);CHKERRQ(ierr);
          	return 121;
            }
          }
		} else if (strdmode==2) {
		  ReadHamInt(inpstr,'>',strdmode,&j,(i-1)*2,matdata.helec);
		  strdmode=0;
		} else if (strdmode==3) {
		  ReadHamInt(inpstr,'|',strdmode,&j,i*2-1,matdata.helec);
		  strdmode=0;
		} else if (curchar=='[') {
		  if (strdmode==1) {
		    j++;
		    ReadHamInt(inpstr,']',strdmode,&j,i-1,matdata.hvalref);
			if (matdata.hvalref[i-1]>vlength){
			  vlength=matdata.hvalref[i-1];
			}
			matdata.hvalref[i-1]=matdata.hvalref[i-1]-1;
		    strdmode=0;
		  } else if (strdmode==4) {
		    j++;
		    ReadHamInt(inpstr,']',strdmode,&j,(i-1)*matdata.nvib,matdata.hvibes);
		    strdmode=0;
		  } else if (strdmode==5) {
		    j++;
		    ReadHamInt(inpstr,']',strdmode,&j,(i-1)*matdata.nvib,matdata.hvibes);
		    strdmode=0;
	      } else if (strdmode==7) {
		    j++;
		    ReadHamInt(inpstr,']',5,&j,(i-1)*matdata.nvib,matdata.hfreq);
		    strdmode=0;
		  } else {
			ierr = PetscPrintf(PETSC_COMM_WORLD,"Error reading Hamiltonian. Unexpected format.\n");CHKERRQ(ierr);
		  }
		}
	  } else {
		break;
	  }
	  j++;
	  curchar=inpstr[j];
	}
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of different constants: %d\n",vlength);CHKERRQ(ierr);
  PetscMalloc(vlength*sizeof(PetscScalar),&matdata.hvalues);
  PetscMalloc(matdata.nvib*sizeof(PetscScalar),&matdata.gsfreq);
  
  while(strcmp(inpstr,"$parameters\n")!=0){
    if(fgets(inpstr,inpstrl,inputfile)==NULL){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nBefore $parameters section.\n");CHKERRQ(ierr);
	  return 101;
    }
  }
  j=0;
  
  /*-----------------------------------------------
   Variables read loop
  -----------------------------------------------*/
  
  if(fscanf(inputfile,"%s",inpstr)==0){
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of file inside $parameters.\n");CHKERRQ(ierr);
    return 101;
  }
  
  while(strcmp(inpstr,"$end")!=0){
	j=0;
	curchar=inpstr[j];
	i=0;
	while((j<inpstrl)&&(i==0)) {
      if(curchar=='[') {
	    j++;
		ReadHamInt(inpstr,']',6,&j,1,&tempint);
		j++;
		ReadInputFloat(inputfile,matdata.hvalues,tempint);
		i=1;
	  }
	  j++;
	  curchar=inpstr[j];
	}
	 if(fscanf(inputfile,"%s",inpstr)==0){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of file inside $parameters.\n");CHKERRQ(ierr);
      return 101;
    }
  } 
  
  while(strcmp(inpstr,"$groundfrequency\n")!=0){
    if(fgets(inpstr,inpstrl,inputfile)==NULL){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of input file.\nBefore $groundfrequency section.\n");CHKERRQ(ierr);
	  return 101;
    }
  }
  j=0;
  
  /*-----------------------------------------------
   groundfrequency frequency read loop
  -----------------------------------------------*/
  
  if(fscanf(inputfile,"%s",inpstr)==0){
	ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of file inside $groundfrequency.\n");CHKERRQ(ierr);
    return 101;
  }
  
  while(strcmp(inpstr,"$end")!=0){
	j=0;
	curchar=inpstr[j];
	i=0;
	while((j<inpstrl)&&(i==0)) {
      if(curchar=='[') {
	    j++;
		ReadHamInt(inpstr,']',6,&j,1,&tempint);
		j++;
		ReadInputFloat(inputfile,matdata.gsfreq,tempint);
		i=1;
	  }
	  j++;
	  curchar=inpstr[j];
	}
	 if(fscanf(inputfile,"%s",inpstr)==0){
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Premature end of file inside $groundfrequency.\n");CHKERRQ(ierr);
      return 101;
    }
  } 
  
  
  fclose(inputfile);
  
  
  
/*  for(i=0;i<matdata.hlength;i++){
    printf("hvalref: %d\n",matdata.hvalref[i]);
  }*/
//  printf("Number of values: %d\n",vlength);
/*  for(i=0;i<vlength;i++){
    printf("values: %f\n",matdata.hvalues[i]);
  }*/
  
  /*----------------------------------------------------------------------
   Set Up markers
  -----------------------------------------------------------------------*/
  matdata.markers[0]=1;
  for(i=1;i<matdata.nvib;i++) {
    matdata.markers[i]=matdata.markers[i-1]*matdata.vbasis[i-1];
  }
  matdata.vsize=matdata.markers[matdata.nvib-1]*matdata.vbasis[matdata.nvib-1];
  matdata.size=matdata.vsize*matdata.nelec;
  
  PetscMalloc(matdata.size*sizeof(PetscScalar),&diagscal);
  PetscMalloc(matdata.size*sizeof(PetscScalar),&outscal);
  /*--------------------------------------------------------------------
   Determine Elements that contribute to diagonal
   -------------------------------------------------------------------*/
   
   for(i=0;i<matdata.hlength;i++){
	 matdata.diag[i]=1;
	 if(matdata.helec[2*i]==matdata.helec[2*i+1]){
	   for(j=0;j<matdata.nvib;j++){
	     if(matdata.hvibes[i*matdata.nvib+j]!=0) {
	       matdata.diag[i]=0;
	     }
	   }
	   if(matdata.diag[i]==1){
	     matdata.diag[i]=2;
		 for(j=0;j<matdata.nvib;j++){
	       if(matdata.hfreq[i*matdata.nvib+j]!=0) {
	         matdata.diag[i]=1;
	       }
	     }
	     
	   }
	 } else {
	   matdata.diag[i]=0;
	 }
   }
/*  for(i=0;i<matdata.hlength;i++){
    printf("diag: %d\n",matdata.diag[i]);
  }*/

/*  for(i=0;i<matdata.nvib;i++){
    printf("markers: %d\n",matdata.markers[i]);
	printf("vbasis: %d\n",matdata.vbasis[i]);
  } */
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, Ax=kx
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  
 
  ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,matdata.size,matdata.size,&matdata,&B);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatShellSetOperation(B,MATOP_MULT,(void(*)())MatrixMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(B,MATOP_MULT_TRANSPOSE,(void(*)())MatrixMult);CHKERRQ(ierr);
  ierr = MatShellSetOperation(B,MATOP_GET_DIAGONAL,(void(*)())MatrixDiagonal);CHKERRQ(ierr);
  
  
  /*get diagonal*/
  
  VecCreate(PETSC_COMM_WORLD,&testvec);
  VecSetSizes(testvec,PETSC_DECIDE,matdata.size);
  VecSetType(testvec, "standard");
  VecCreate(PETSC_COMM_WORLD,&testvec2);
  VecSetSizes(testvec2,PETSC_DECIDE,matdata.size);
  VecSetType(testvec2, "standard");
  /*
  ierr = PetscPrintf(PETSC_COMM_WORLD,"diagonal\n");CHKERRQ(ierr);
  MatrixDiagonal(B,testvec2);
  VecView(testvec2,PETSC_VIEWER_STDOUT_WORLD);
  MPI_Barrier(PETSC_COMM_WORLD);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Matrix\n");CHKERRQ(ierr);
  
  
  for(i=0;i<matdata.size;i++){
    if(i>0){
	  ierr = VecSetValue(testvec,i-1,0,INSERT_VALUES);CHKERRQ(ierr);
	}
    ierr = VecSetValue(testvec,i,1,INSERT_VALUES);CHKERRQ(ierr);
    VecAssemblyBegin(testvec);
    VecAssemblyEnd(testvec);
    MatrixMult(B,testvec,testvec2);
	VecView(testvec,PETSC_VIEWER_STDOUT_WORLD);
    VecView(testvec2,PETSC_VIEWER_STDOUT_WORLD);
  }
  */
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create eigensolver context
  */
  ierr = EPSCreate(PETSC_COMM_WORLD,&eps);CHKERRQ(ierr);

  /*
     Set operators. In this case, it is a standard eigenvalue problem
  */
  ierr = EPSSetOperators(eps,B,NULL);CHKERRQ(ierr);
  ierr = EPSSetProblemType(eps,EPS_HEP);CHKERRQ(ierr);

  /*
     Set solver parameters at runtime
  */
  ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
    
  ierr = EPSSetDimensions(eps,nev,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = EPSSetWhichEigenpairs(eps,	EPS_SMALLEST_REAL);CHKERRQ(ierr);
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                      Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = EPSSolve(eps);CHKERRQ(ierr);
  

  /*
     Optional: Get some information from the solver and display it
  */
  ierr = EPSGetType(eps,&type);CHKERRQ(ierr);  
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solution method: %s\n\n",type);CHKERRQ(ierr);
  ierr = EPSGetDimensions(eps,&nev,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of requested eigenvalues: %D\n",nev);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                    Display solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = EPSPrintSolution(eps,NULL);CHKERRQ(ierr);
  
  ierr = EPSGetConverged(eps,&nconv);
  
  if(nconv>matdata.vsize){
    nconv=matdata.vsize;
  }
  
  for(i=0;i<matdata.nvib;i++){
    colbf[i]=1;
  }
  halfw=vecdot(matdata.nvib,matdata.gsfreq,colbf);
    
//  if(rank==0){
//  VecCreate(PETSC_COMM_WORLD,&outfullvec);
//  VecSetType(outfullvec, "seq");
//  VecSetSizes(outfullvec,matdata.size,matdata.size);
//  }
    
//  ierr = VecGetSize(outfullvec,&tempint);CHKERRQ(ierr);
//  ierr = PetscPrintf(PETSC_COMM_WORLD,"Size of output vector: %d\n",tempint);
  
  for(i=0;i<nconv;i++){
    ierr = EPSGetEigenvector(eps,i,testvec,testvec2);CHKERRQ(ierr);
		
	VecScatterCreateToZero(testvec,&collectctx,&outfullvec);
    // scatter as many times as you need
    VecScatterBegin(collectctx,testvec,outfullvec,INSERT_VALUES,SCATTER_FORWARD);
    VecScatterEnd(collectctx,testvec,outfullvec,INSERT_VALUES,SCATTER_FORWARD);
    // destroy scatter context and local vector when no longer needed
    VecScatterDestroy(&collectctx);
	

	/*begin rank 0 print statement*/
    if(rank==0){
	ierr = VecGetArray(outfullvec,&outscal);CHKERRQ(ierr);
	ierr = EPSGetEigenvalue(eps,i,&eval,&tempscal);CHKERRQ(ierr);
	
	if(i==0){
	  leval=eval;
	}
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%10.3f",eval-leval);CHKERRQ(ierr);
    for(j=0;j<matdata.nelec;j++){
      tempint=j*matdata.vsize;
	  tempscal=outscal[tempint];
      ierr = PetscPrintf(PETSC_COMM_WORLD,"    %11.8f",tempscal);CHKERRQ(ierr);
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
  
    strcpy(inpstr,"outspec_");
    sprintf(&inpstr[8],"%d",i);
    strcat(inpstr,".txt");
    curoutfile=fopen(inpstr,"w");
    fprintf(curoutfile,"Frequency  Values\n");
    complete=0;
    while(complete==0){
      curind=vecmarkers(matdata.nvib,colbf,matdata.markers);
	  if(vecdot(matdata.nvib,matdata.gsfreq,colbf)-halfw<range){
	    fprintf(curoutfile,"%10.3f",vecdot(matdata.nvib,matdata.gsfreq,colbf)-halfw);
	    for(j=0;j<matdata.nelec;j++){
	      fprintf(curoutfile,"   %11.8f",outscal[curind+j*matdata.vsize]);
	    }
	    fprintf(curoutfile,"\n");
	  }
      actvib=0;
      colbf[actvib]=colbf[actvib]+1;
	  for(j=0;j<matdata.nvib;j++){
	    if(colbf[j]>matdata.vbasis[j]){
	      colbf[j]=1;
	  	if(j<matdata.nvib-1){
	  	  colbf[j+1]=colbf[j+1]+1;
		} else {
		  complete=1;
		}
	  }
	}
  }
  
  
  fclose(curoutfile);
  ierr = VecRestoreArray(testvec,&outscal);CHKERRQ(ierr);
  }
  }
  /*End rank 0 print statement*/
  ierr = EPSDestroy(&eps);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = SlepcFinalize();
  return 0;
}

/*
    Compute the matrix vector multiplication y<---T*x where T is a nx by nx
    tridiagonal matrix with DD on the diagonal, DL on the subdiagonal, and
    DU on the superdiagonal.
 */

PetscErrorCode MatrixMult(Mat A,Vec x,Vec y)
{
  PetscErrorCode    ierr;
  MPI_Status        MPIstatus;
  struct data       matdata;
  void              *ctx;
  PetscInt          i,j,actvib,curind,rowind,remainder,ngo,nodeloop,colelecbf,rowelecbf;
  PetscInt          *colbf,*rowbf,maxsize;
  PetscInt          tempint;
  const PetscInt    *ownerv;
  PetscMPIInt       size,rank,nodesend;
  const PetscScalar *pin;
  PetscScalar       *pout,*psend,*precv;
  int               complete=0;
//  FILE              *poutfile;
//  char              outstr[20];
//  PetscFunctionBeginUser;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  /*  Allocate memory for vector ranges*/
//  PetscMalloc((size+1)*sizeof(PetscInt),&ownerv);
  /*  Get ownership ranges*/
  ierr = VecGetOwnershipRanges(y,&ownerv);CHKERRQ(ierr);
  
  /*node file*/
  
//  strcpy(outstr,"pout_mult_");
//  sprintf(&outstr[10],"%d",rank);
//  strcat(outstr,".txt");
//  poutfile=fopen(outstr,"w");
//  fprintf(poutfile,"Out multiply file for: %d\n",rank);
//  fprintf(poutfile,"Value of start: %d end: %d\n",ownerv[rank],ownerv[rank+1]);
  

  /* get matrix data*/
  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);
  matdata = *(struct data*)ctx;
  
  /*  Allocate memory for counters*/
  PetscMalloc(matdata.nvib*sizeof(PetscInt),&colbf);
  PetscMalloc(matdata.nvib*sizeof(PetscInt),&rowbf);
  
  /*Figure out which node has the biggest vector*/
  maxsize=0;
  for(i=0;i<size;i++){
    if((ownerv[i+1]-ownerv[i])>maxsize){
	  maxsize=(ownerv[i+1]-ownerv[i]);
	}
  }
//  fprintf(poutfile,"Max size: %d\n",maxsize);
  PetscMalloc(maxsize*sizeof(PetscScalar),&psend);
  PetscMalloc(maxsize*sizeof(PetscScalar),&precv);
  /*Get vectors from Petsc*/
  ierr = VecGetArrayRead(x,&pin);CHKERRQ(ierr);
  ierr = VecGetArray(y,&pout);CHKERRQ(ierr);
  
//  ierr = VecGetLocalSize(y,&tempint);CHKERRQ(ierr);
//  ierr = fprintf(poutfile,"Size of pout: %d\n",tempint);CHKERRQ(ierr);
  /*Zero out the vectors*/
  for(i=0;i<(ownerv[rank+1]-ownerv[rank]);i++){
    pout[i]=0;
  }
  
  /*loop over nodes to send to*/
  for(nodeloop=0;nodeloop<size;nodeloop++){
//    fprintf(poutfile,"**************\nBegin Node loop\n**************\n");
    nodesend=rank ^ nodeloop;
//	fprintf(poutfile,"Send to: %d\n",nodesend);
	/* Set colbf to start location */
    colelecbf=ownerv[rank]/matdata.vsize+1;
    remainder=ownerv[rank]%matdata.vsize;
	/*reset MPI vectors*/
    for(i=0;i<maxsize;i++){
      psend[i]=0;
      precv[i]=0;
    }
    for(i=0;i<matdata.nvib;i++){
      colbf[matdata.nvib-i-1]=remainder/matdata.markers[matdata.nvib-i-1]+1;
      remainder=remainder%matdata.markers[matdata.nvib-i-1];
    }
    /*loop over owned input vector*/
    for(curind=ownerv[rank];curind<ownerv[rank+1];curind++){
	  /*print current element*/
//      fprintf(poutfile,"elec: %d colbf: ",colelecbf);
//      for(i=0;i<matdata.nvib;i++){
//        fprintf(poutfile," %d ",colbf[i]);
//  	  }
//      fprintf(poutfile,"\n");
	  
	  /*loop over Hamiltonian elements*/
      for(i=0;i<matdata.hlength;i++){
	    /*Check if electric operators are compatible with current function*/
	    if(matdata.helec[2*i+1]==colelecbf){
		  /*compute rowbf*/
          vecsum(matdata.nvib,colbf,&matdata.hvibes[i*matdata.nvib],rowbf);
          ngo=0;
	      /*Check if output basis function exists*/
          for(j=0;j<matdata.nvib;j++){
	        if((rowbf[j]<1)||(rowbf[j]>matdata.vbasis[j])){
	          ngo=1;
		    }
	      }
	      if(ngo==0){
		    /* Compute out row index*/
	        rowind=vecmarkers(matdata.nvib,rowbf,matdata.markers)+(matdata.helec[2*i]-1)*matdata.vsize;
			/*Check if it is in output range*/
            if((rowind>=ownerv[nodesend])&&(rowind<ownerv[nodesend+1])){
			  
//			  fprintf(poutfile,"Editing location of psend: %d\n",rowind-ownerv[nodesend]);
//			  fprintf(poutfile,"pout: %f\n",psend[rowind-ownerv[nodesend]]);
//			  fprintf(poutfile,"Accessing location of pin: %d\n",curind-ownerv[rank]);
//			  fprintf(poutfile,"pin: %f\n",pin[curind-ownerv[rank]]);
//			  fprintf(poutfile,"Hamiltonian element: %d\n",i);
//			  fprintf(poutfile,"Hamiltonian element contribution: %d\n",i);
//			  fprintf(poutfile,"Value: %f\n",pin[curind-ownerv[rank]]*(matdata.hvalues[matdata.hvalref[i]]*vecpowers(matdata.nvib,colbf,&matdata.hfreq[i*matdata.nvib])*raiselower(matdata.nvib,colbf,&matdata.hvibes[i*matdata.nvib])));
			  
		      /*compute contribution*/
			  psend[rowind-ownerv[nodesend]]+=pin[curind-ownerv[rank]]*(matdata.hvalues[matdata.hvalref[i]]*vecpowers(matdata.nvib,colbf,&matdata.hfreq[i*matdata.nvib])*raiselower(matdata.nvib,colbf,&matdata.hvibes[i*matdata.nvib]));
//			  fprintf(poutfile,"location edited\n\n");
		    }
          }
   	    }
      }
    
      actvib=0;
      colbf[actvib]=colbf[actvib]+1;
  	  for(i=0;i<matdata.nvib;i++){
  	    if(colbf[i]>matdata.vbasis[i]){
  	      colbf[i]=1;
          if(i<matdata.nvib-1){
            colbf[i+1]=colbf[i+1]+1;
          } else {
            colelecbf=colelecbf+1;
          }
        }
  	  }
    }
	
    if(nodeloop==0){
      for(i=0;i<(ownerv[rank+1]-ownerv[rank]);i++){
	    pout[i]=psend[i];
	  }
    }else{
//	  fprintf(poutfile,"Prepairing to send\n");
      MPI_Sendrecv( psend , (ownerv[nodesend+1]-ownerv[nodesend]) , MPIU_SCALAR , nodesend , nodeloop ,
	                precv , (ownerv[rank+1]-ownerv[rank]) , MPIU_SCALAR , nodesend , nodeloop ,
				    MPI_COMM_WORLD , &MPIstatus);
//	  fprintf(poutfile,"Send complete\n");
      for(i=0;i<(ownerv[rank+1]-ownerv[rank]);i++){
	    pout[i]+=precv[i];
//		fprintf(poutfile,"Editing values %d of pout.\n",i);
	  }
    }	
  }
//  fprintf(poutfile,"Restoring Values\n");
  ierr = VecRestoreArrayRead(x,&pin);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&pout);CHKERRQ(ierr);
  ierr = PetscFree(psend);CHKERRQ(ierr);
  ierr = PetscFree(precv);CHKERRQ(ierr);
  ierr = PetscFree(colbf);CHKERRQ(ierr);
  ierr = PetscFree(rowbf);CHKERRQ(ierr);
//  ierr = PetscFree(ownerv);CHKERRQ(ierr);
//  fprintf(poutfile,"Values restored, returning\n");
//  fclose(poutfile);
  PetscFunctionReturn(0);
}

PetscErrorCode MatrixDiagonal(Mat A,Vec diag)
{
  PetscErrorCode    ierr;
  struct data       matdata;
  void              *ctx;
  PetscInt          i,j,actvib,curind;
  PetscInt          *colbf,elecbf,istart,iend,remainder;
  PetscMPIInt       size,rank;
  PetscScalar       *pdiag;
  int               complete=0;
  /*
  FILE              *poutfile;
  char              outstr[20];
  */
  
//  PetscFunctionBeginUser;
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  
  ierr = VecGetOwnershipRange(diag,&istart,&iend);CHKERRQ(ierr);
  
  /*
  strcpy(outstr,"pout_");
  sprintf(&outstr[5],"%d",rank);
  strcat(outstr,".txt");
  poutfile=fopen(outstr,"w");
  fprintf(poutfile,"Out file for: %d\n",rank);
  fprintf(poutfile,"Value of start: %d end: %d\n",istart,iend);
  */
  ierr = MatShellGetContext(A,&ctx);CHKERRQ(ierr);
  matdata = *(struct data*)ctx;
  
  
  /*  Allocate memory for counters*/
  PetscMalloc(matdata.nvib*sizeof(PetscInt),&colbf);
  /*Get vectors from Petsc*/
  ierr = VecGetArray(diag,&pdiag);CHKERRQ(ierr);
  
  /* Set colbf to istart location */
  elecbf=istart/matdata.vsize+1;
  remainder=istart%matdata.vsize;
  for(i=0;i<matdata.nvib;i++){
    colbf[matdata.nvib-i-1]=remainder/matdata.markers[matdata.nvib-i-1]+1;
    remainder=remainder%matdata.markers[matdata.nvib-i-1];
  }
  
  for(j=istart;j<iend;j++){
    /*
    fprintf(poutfile,"elec: %d colbf: ",elecbf);
	for(i=0;i<matdata.nvib;i++){
	  fprintf(poutfile," %d ",colbf[i]);
	}
	fprintf(poutfile,"\n");
	*/
    curind=j;
    for(i=0;i<matdata.hlength;i++){
	  if(matdata.diag[i]>0){
/*	    printf("diag ind: %d h-value: %f vecpowers: %f\n",curind+(matdata.helec[i*2+1]-1)*matdata.vsize,matdata.hvalues[matdata.hvalref[i]],vecpowers(matdata.nvib,colbf,&matdata.hfreq[i*matdata.nvib]));*/
        if(matdata.helec[i*2+1]==elecbf){
	      pdiag[j-istart]+=(matdata.hvalues[matdata.hvalref[i]]*vecpowers(matdata.nvib,colbf,&matdata.hfreq[i*matdata.nvib]));
        }
	  }
	}
  
    actvib=0;
    colbf[actvib]=colbf[actvib]+1;
	for(i=0;i<matdata.nvib;i++){
	  if(colbf[i]>matdata.vbasis[i]){
	    colbf[i]=1;
		if(i<matdata.nvib-1){
		  colbf[i+1]=colbf[i+1]+1;
		} else {
		  elecbf=elecbf+1;
		}
	  }
	}
  }
  
  ierr = VecRestoreArray(diag,&pdiag);CHKERRQ(ierr);
  ierr = PetscFree(colbf);CHKERRQ(ierr);
  VecAssemblyBegin(diag);
  VecAssemblyEnd(diag);
  PetscFunctionReturn(0);
}

static void vecsum(const int n,const PetscInt *x, const PetscInt *y,PetscInt *ret)
{
  int i;
  for(i=0;i<n;i++){
    ret[i]=x[i]+y[i];
  }
}

static void vecprod(const int n, const PetscInt *x, const PetscInt *y,PetscInt *ret)
{
  int i;
  for(i=0;i<n;i++){
    ret[i]=x[i]*y[i];
  }
}

PetscScalar vecdot(const int n, const PetscScalar *x, const PetscInt *y)
{
  PetscInt i;
  PetscScalar ret=0;
  for(i=0;i<n;i++){
    ret=ret+x[i]*y[i];
  }
  return(ret);
}

PetscInt vecmarkers(const int n, const PetscInt *x, const PetscInt *y)
{
  PetscInt i,ret=0;
  for(i=0;i<n;i++){
    ret=ret+(x[i]-1)*y[i];
  }
  return(ret);
}

PetscScalar vecpowers(const int n, const PetscInt *vib, const PetscInt *order){
  PetscInt i;
  PetscScalar ret=1;
  for(i=0;i<n;i++){
/*    printf("vecpowers: vib: %d order: %d result %f\n",vib[i],order[i],pow(vib[i],order[i]));*/
    ret=ret*(PetscInt) pow(vib[i],order[i]);
  }
  return(ret);
}

PetscScalar raiselower(const int n,const PetscInt *colbf,const PetscInt *opp){
  PetscInt      i,j;
  PetscScalar   ret=1;
  
  for(i=0;i<n;i++){
    if(opp[i]>0){
/*	  printf("Computing raising operator\n");*/
      for(j=0;j<opp[i];j++){
	    ret=ret*sqrt(colbf[i]+j);
	  }
/*	  printf("value of raising: %f\n",ret);*/
	} else if(opp[i]<0){
	/*printf("Computing lowering operator\n");
	  printf("opp: %d colbf: %d index: %d\n",opp[i],colbf[i],i);*/
      for(j=0;j>opp[i];j--){
	    ret=ret*sqrt(colbf[i]+j-1);
	  }
/*	  printf("value of lowering: %f\n",ret);*/
	}
  }
  return(ret);
}


int ReadInputInt(FILE *inputfile,PetscInt *variable,int mode) {
  char           str1[100];
  
  if(fscanf(inputfile,"%s",str1)==0){
    puts("ReadInputInt error:");
	return 1001;
  } else {
	*variable=atoi(str1);
  }
  if(mode==0) {
    if(fgets(str1,100,inputfile)==NULL){
      puts("ReadInputInt error:");
      return 1011;
    }
  }
  return 0;
}

int ReadInputFloat(FILE *inputfile,PetscScalar *variable,PetscInt index) {
  char           str1[100];

  if(fscanf(inputfile,"%s",str1)==0){
	return 1001;
  } else {
	variable[index-1]=atof(str1);
  }
  if(fgets(str1,100,inputfile)==NULL){
    puts("ReadInputFloat error:");
	puts("Reading \\n");
    return 1011;
  }
  return 0;
}

int ReadHamInt(char *inpstr,char termchar,PetscInt strdmode, PetscInt *j,PetscInt loc,PetscInt *variable){
  const PetscInt inpstrl=200;
  char curchar;
  char tmpstr[10];
  int  k=0,tempint;
  
  strcpy(tmpstr,"          ");
  curchar=inpstr[(*j)];
  while(curchar!=termchar) {
    tmpstr[k]=curchar;
	k++;
	(*j)++;
	curchar=inpstr[*j];
	if((*j)>inpstrl){
	  puts("Error reading Hamiltonian.");
	  puts("Unclosed [.");
	  return 1021;
	}
  }
  tempint=atoi(tmpstr);
  /* printf("tempint: %d\n",tempint);
  printf("tmpstr: %s\n",tmpstr);
  printf("strdmode: %d\n",strdmode);
  printf("location: %d\n",loc);*/
  if(tempint==0){
    puts("Error reading Hamiltonian.");
	printf("Could not parse integer: %s\n",tmpstr);
	printf("String: %s; j val: %d; k val: %d\n",inpstr,*j,k);
	return 1022;
  }
  if(strdmode==4){
    variable[loc+tempint-1]--;
  } else if(strdmode==5){
    variable[loc+tempint-1]++;
  } else if(strdmode==6){
    *variable=tempint;
  } else {
    variable[loc]=tempint;
  }
  return 0;
  
}
  
 
