      SUBROUTINE RADAU5(N,FCN,X,Y,XEND,H,
     &                  RTOL,ATOL,ITOL,
     &                  JAC ,IJAC,MLJAC,MUJAC,
     &                  MAS ,IMAS,MLMAS,MUMAS, var_index,
     &                  SOLOUT,REPORTFUN,nReport,IOUT,
     &                  WORK,LWORK,IWORK,LIWORK,RPAR,IPAR,IDID,
     &                  nPrint, nMaxBadIte, nAlwaysUse2ndErrorEstimate
     &                  )
C ----------------------------------------------------------
C     NUMERICAL SOLUTION OF A STIFF (OR DIFFERENTIAL ALGEBRAIC)
C     SYSTEM OF FIRST 0RDER ORDINARY DIFFERENTIAL EQUATIONS
C                     M*Y'=F(X,Y).
C     THE SYSTEM CAN BE (LINEARLY) IMPLICIT (MASS-MATRIX M .NE. I)
C     OR EXPLICIT (M=I).
C     THE METHOD USED IS AN IMPLICIT RUNGE-KUTTA METHOD (RADAU IIA)
C     OF ORDER 5 WITH STEP SIZE CONTROL AND CONTINUOUS OUTPUT.
C     CF. SECTION IV.8
C
C     AUTHORS: E. HAIRER AND G. WANNER
C              UNIVERSITE DE GENEVE, DEPT. DE MATHEMATIQUES
C              CH-1211 GENEVE 24, SWITZERLAND 
C              E-MAIL:  Ernst.Hairer@math.unige.ch
C                       Gerhard.Wanner@math.unige.ch
C     
C     THIS CODE IS PART OF THE BOOK:
C         E. HAIRER AND G. WANNER, SOLVING ORDINARY DIFFERENTIAL
C         EQUATIONS II. STIFF AND DIFFERENTIAL-ALGEBRAIC PROBLEMS.
C         SPRINGER SERIES IN COMPUTATIONAL MATHEMATICS 14,
C         SPRINGER-VERLAG 1991, SECOND EDITION 1996.
C      
C     VERSION OF JULY 9, 1996
C     (latest small correction: January 18, 2002)
C
C     INPUT PARAMETERS  
C     ----------------  
C     N           DIMENSION OF THE SYSTEM 
C
C     FCN         NAME (EXTERNAL) OF SUBROUTINE COMPUTING THE
C                 VALUE OF F(X,Y):
C                    SUBROUTINE FCN(N,X,Y,F,RPAR,IPAR)
C                    DOUBLE PRECISION X,Y(N),F(N)
C                    F(1)=...   ETC.
C                 RPAR, IPAR (SEE BELOW)
C
C     X           INITIAL X-VALUE
C
C     Y(N)        INITIAL VALUES FOR Y
C
C     XEND        FINAL X-VALUE (XEND-X MAY BE POSITIVE OR NEGATIVE)
C
C     H           INITIAL STEP SIZE GUESS;
C                 FOR STIFF EQUATIONS WITH INITIAL TRANSIENT, 
C                 H=1.D0/(NORM OF F'), USUALLY 1.D-3 OR 1.D-5, IS GOOD.
C                 THIS CHOICE IS NOT VERY IMPORTANT, THE STEP SIZE IS
C                 QUICKLY ADAPTED. (IF H=0.D0, THE CODE PUTS H=1.D-6).
C
C     RTOL,ATOL   RELATIVE AND ABSOLUTE ERROR TOLERANCES. THEY
C                 CAN BE BOTH SCALARS OR ELSE BOTH VECTORS OF LENGTH N.
C
C     ITOL        SWITCH FOR RTOL AND ATOL:
C                   ITOL=0: BOTH RTOL AND ATOL ARE SCALARS.
C                     THE CODE KEEPS, ROUGHLY, THE LOCAL ERROR OF
C                     Y(I) BELOW RTOL*ABS(Y(I))+ATOL
C                   ITOL=1: BOTH RTOL AND ATOL ARE VECTORS.
C                     THE CODE KEEPS THE LOCAL ERROR OF Y(I) BELOW
C                     RTOL(I)*ABS(Y(I))+ATOL(I).
C
C     JAC         NAME (EXTERNAL) OF THE SUBROUTINE WHICH COMPUTES
C                 THE PARTIAL DERIVATIVES OF F(X,Y) WITH RESPECT TO Y
C                 (THIS ROUTINE IS ONLY CALLED IF IJAC=1; SUPPLY
C                 A DUMMY SUBROUTINE IN THE CASE IJAC=0).
C                 FOR IJAC=1, THIS SUBROUTINE MUST HAVE THE FORM
C                    SUBROUTINE JAC(N,X,Y,DFY,LDFY,RPAR,IPAR)
C                    DOUBLE PRECISION X,Y(N),DFY(LDFY,N)
C                    DFY(1,1)= ...
C                 LDFY, THE COLUMN-LENGTH OF THE ARRAY, IS
C                 FURNISHED BY THE CALLING PROGRAM.
C                 IF (MLJAC.EQ.N) THE JACOBIAN IS SUPPOSED TO
C                    BE FULL AND THE PARTIAL DERIVATIVES ARE
C                    STORED IN DFY AS
C                       DFY(I,J) = PARTIAL F(I) / PARTIAL Y(J)
C                 ELSE, THE JACOBIAN IS TAKEN AS BANDED AND
C                    THE PARTIAL DERIVATIVES ARE STORED
C                    DIAGONAL-WISE AS
C                       DFY(I-J+MUJAC+1,J) = PARTIAL F(I) / PARTIAL Y(J).
C
C     IJAC        SWITCH FOR THE COMPUTATION OF THE JACOBIAN:
C                    IJAC=0: JACOBIAN IS COMPUTED INTERNALLY BY FINITE
C                       DIFFERENCES, SUBROUTINE "JAC" IS NEVER CALLED.
C                    IJAC=1: JACOBIAN IS SUPPLIED BY SUBROUTINE JAC.
C
C     MLJAC       SWITCH FOR THE BANDED STRUCTURE OF THE JACOBIAN:
C                    MLJAC=N: JACOBIAN IS A FULL MATRIX. THE LINEAR
C                       ALGEBRA IS DONE BY FULL-MATRIX GAUSS-ELIMINATION.
C                    0<=MLJAC<N: MLJAC IS THE LOWER BANDWITH OF JACOBIAN 
C                       MATRIX (>= NUMBER OF NON-ZERO DIAGONALS BELOW
C                       THE MAIN DIAGONAL).
C
C     MUJAC       UPPER BANDWITH OF JACOBIAN  MATRIX (>= NUMBER OF NON-
C                 ZERO DIAGONALS ABOVE THE MAIN DIAGONAL).
C                 NEED NOT BE DEFINED IF MLJAC=N.
C
C     ----   MAS,IMAS,MLMAS, AND MUMAS HAVE ANALOG MEANINGS      -----
C     ----   FOR THE "MASS MATRIX" (THE MATRIX "M" OF SECTION IV.8): -
C
C     MAS         NAME (EXTERNAL) OF SUBROUTINE COMPUTING THE MASS-
C                 MATRIX M.
C                 IF IMAS=0, THIS MATRIX IS ASSUMED TO BE THE IDENTITY
C                 MATRIX AND NEEDS NOT TO BE DEFINED;
C                 SUPPLY A DUMMY SUBROUTINE IN THIS CASE.
C                 IF IMAS=1, THE SUBROUTINE MAS IS OF THE FORM
C                    SUBROUTINE MAS(N,AM,LMAS,RPAR,IPAR)
C                    DOUBLE PRECISION AM(LMAS,N)
C                    AM(1,1)= ....
C                    IF (MLMAS.EQ.N) THE MASS-MATRIX IS STORED
C                    AS FULL MATRIX LIKE
C                         AM(I,J) = M(I,J)
C                    ELSE, THE MATRIX IS TAKEN AS BANDED AND STORED
C                    DIAGONAL-WISE AS
C                         AM(I-J+MUMAS+1,J) = M(I,J)
C
C     IMAS       GIVES INFORMATION ON THE MASS-MATRIX:
C                    IMAS=0: M IS SUPPOSED TO BE THE IDENTITY
C                       MATRIX, MAS IS NEVER CALLED.
C                    IMAS=1: MASS-MATRIX  IS SUPPLIED.
C
C     MLMAS       SWITCH FOR THE BANDED STRUCTURE OF THE MASS-MATRIX:
C                    MLMAS=N: THE FULL MATRIX CASE. THE LINEAR
C                       ALGEBRA IS DONE BY FULL-MATRIX GAUSS-ELIMINATION.
C                    0<=MLMAS<N: MLMAS IS THE LOWER BANDWITH OF THE
C                       MATRIX (>= NUMBER OF NON-ZERO DIAGONALS BELOW
C                       THE MAIN DIAGONAL).
C                 MLMAS IS SUPPOSED TO BE .LE. MLJAC.
C
C     MUMAS       UPPER BANDWITH OF MASS-MATRIX (>= NUMBER OF NON-
C                 ZERO DIAGONALS ABOVE THE MAIN DIAGONAL).
C                 NEED NOT BE DEFINED IF MLMAS=N.
C                 MUMAS IS SUPPOSED TO BE .LE. MUJAC.
C
C     SOLOUT      NAME (EXTERNAL) OF SUBROUTINE PROVIDING THE
C                 NUMERICAL SOLUTION DURING INTEGRATION. 
C                 IF IOUT=1, IT IS CALLED AFTER EVERY SUCCESSFUL STEP.
C                 SUPPLY A DUMMY SUBROUTINE IF IOUT=0. 
C                 IT MUST HAVE THE FORM
C                    SUBROUTINE SOLOUT (NR,XOLD,X,Y,CONT,LRC,N,
C                                       RPAR,IPAR,IRTRN)
C                    DOUBLE PRECISION X,Y(N),CONT(LRC)
C                    ....  
C                 SOLOUT FURNISHES THE SOLUTION "Y" AT THE NR-TH
C                    GRID-POINT "X" (THEREBY THE INITIAL VALUE IS
C                    THE FIRST GRID-POINT).
C                 "XOLD" IS THE PRECEEDING GRID-POINT.
C                 "IRTRN" SERVES TO INTERRUPT THE INTEGRATION. IF IRTRN
C                    IS SET <0, RADAU5 RETURNS TO THE CALLING PROGRAM.
C           
C          -----  CONTINUOUS OUTPUT: -----
C                 DURING CALLS TO "SOLOUT", A CONTINUOUS SOLUTION
C                 FOR THE INTERVAL [XOLD,X] IS AVAILABLE THROUGH
C                 THE FUNCTION
C                        >>>   CONTR5(I,S,CONT,LRC)   <<<
C                 WHICH PROVIDES AN APPROXIMATION TO THE I-TH
C                 COMPONENT OF THE SOLUTION AT THE POINT S. THE VALUE
C                 S SHOULD LIE IN THE INTERVAL [XOLD,X].
C                 DO NOT CHANGE THE ENTRIES OF CONT(LRC), IF THE
C                 DENSE OUTPUT FUNCTION IS USED.
C
C     IOUT        SWITCH FOR CALLING THE SUBROUTINE SOLOUT:
C                    IOUT=0: SUBROUTINE IS NEVER CALLED
C                    IOUT=1: SUBROUTINE IS AVAILABLE FOR OUTPUT.
C
C     WORK        ARRAY OF WORKING SPACE OF LENGTH "LWORK".
C                 WORK(1), WORK(2),.., WORK(20) SERVE AS PARAMETERS
C                 FOR THE CODE. FOR STANDARD USE OF THE CODE
C                 WORK(1),..,WORK(20) MUST BE SET TO ZERO BEFORE
C                 CALLING. SEE BELOW FOR A MORE SOPHISTICATED USE.
C                 WORK(21),..,WORK(LWORK) SERVE AS WORKING SPACE
C                 FOR ALL VECTORS AND MATRICES.
C                 "LWORK" MUST BE AT LEAST
C                             N*(LJAC+LMAS+3*LE+12)+20
C                 WHERE
C                    LJAC=N              IF MLJAC=N (FULL JACOBIAN)
C                    LJAC=MLJAC+MUJAC+1  IF MLJAC<N (BANDED JAC.)
C                 AND                  
C                    LMAS=0              IF IMAS=0
C                    LMAS=N              IF IMAS=1 AND MLMAS=N (FULL)
C                    LMAS=MLMAS+MUMAS+1  IF MLMAS<N (BANDED MASS-M.)
C                 AND
C                    LE=N               IF MLJAC=N (FULL JACOBIAN)
C                    LE=2*MLJAC+MUJAC+1 IF MLJAC<N (BANDED JAC.)
C
C                 IN THE USUAL CASE WHERE THE JACOBIAN IS FULL AND THE
C                 MASS-MATRIX IS THE INDENTITY (IMAS=0), THE MINIMUM
C                 STORAGE REQUIREMENT IS 
C                             LWORK = 4*N*N+12*N+20.
C                 IF IWORK(9)=M1>0 THEN "LWORK" MUST BE AT LEAST
C                          N*(LJAC+12)+(N-M1)*(LMAS+3*LE)+20
C                 WHERE IN THE DEFINITIONS OF LJAC, LMAS AND LE THE
C                 NUMBER N CAN BE REPLACED BY N-M1.
C
C     LWORK       DECLARED LENGTH OF ARRAY "WORK".
C
C     IWORK       INTEGER WORKING SPACE OF LENGTH "LIWORK".
C                 IWORK(1),IWORK(2),...,IWORK(20) SERVE AS PARAMETERS
C                 FOR THE CODE. FOR STANDARD USE, SET IWORK(1),..,
C                 IWORK(20) TO ZERO BEFORE CALLING.
C                 IWORK(21),...,IWORK(LIWORK) SERVE AS WORKING AREA.
C                 "LIWORK" MUST BE AT LEAST 3*N+20.
C
C     LIWORK      DECLARED LENGTH OF ARRAY "IWORK".
C
C     RPAR, IPAR  REAL AND INTEGER PARAMETERS (OR PARAMETER ARRAYS) WHICH  
C                 CAN BE USED FOR COMMUNICATION BETWEEN YOUR CALLING
C                 PROGRAM AND THE FCN, JAC, MAS, SOLOUT SUBROUTINES. 
C
C ----------------------------------------------------------------------
C 
C     SOPHISTICATED SETTING OF PARAMETERS
C     -----------------------------------
C              SEVERAL PARAMETERS OF THE CODE ARE TUNED TO MAKE IT WORK 
C              WELL. THEY MAY BE DEFINED BY SETTING WORK(1),...
C              AS WELL AS IWORK(1),... DIFFERENT FROM ZERO.
C              FOR ZERO INPUT, THE CODE CHOOSES DEFAULT VALUES:
C
C    IWORK(1)  IF IWORK(1).NE.0, THE CODE TRANSFORMS THE JACOBIAN
C              MATRIX TO HESSENBERG FORM. THIS IS PARTICULARLY
C              ADVANTAGEOUS FOR LARGE SYSTEMS WITH FULL JACOBIAN.
C              IT DOES NOT WORK FOR BANDED JACOBIAN (MLJAC<N)
C              AND NOT FOR IMPLICIT SYSTEMS (IMAS=1).
C
C    IWORK(2)  THIS IS THE MAXIMAL NUMBER OF ALLOWED STEPS.
C              THE DEFAULT VALUE (FOR IWORK(2)=0) IS 100000.
C
C    IWORK(3)  THE MAXIMUM NUMBER OF NEWTON ITERATIONS FOR THE
C              SOLUTION OF THE IMPLICIT SYSTEM IN EACH STEP.
C              THE DEFAULT VALUE (FOR IWORK(3)=0) IS 7.
C
C    IWORK(4)  IF IWORK(4).EQ.0 THE EXTRAPOLATED COLLOCATION SOLUTION
C              IS TAKEN AS STARTING VALUE FOR NEWTON'S METHOD.
C              IF IWORK(4).NE.0 ZERO STARTING VALUES ARE USED.
C              THE LATTER IS RECOMMENDED IF NEWTON'S METHOD HAS
C              DIFFICULTIES WITH CONVERGENCE (THIS IS THE CASE WHEN
C              NSTEP IS LARGER THAN NACCPT + NREJCT; SEE OUTPUT PARAM.).
C              DEFAULT IS IWORK(4)=0.
C
C       THE FOLLOWING 3 PARAMETERS ARE IMPORTANT FOR
C       DIFFERENTIAL-ALGEBRAIC SYSTEMS OF INDEX > 1.
C       THE FUNCTION-SUBROUTINE SHOULD BE WRITTEN SUCH THAT
C       THE INDEX 1,2,3 VARIABLES APPEAR IN THIS ORDER. 
C       IN ESTIMATING THE ERROR THE INDEX 2 VARIABLES ARE
C       MULTIPLIED BY H, THE INDEX 3 VARIABLES BY H**2.
C
C    IWORK(5)  DIMENSION OF THE INDEX 1 VARIABLES (MUST BE > 0). FOR 
C              ODE'S THIS EQUALS THE DIMENSION OF THE SYSTEM.
C              DEFAULT IWORK(5)=N.
C
C    IWORK(6)  DIMENSION OF THE INDEX 2 VARIABLES. DEFAULT IWORK(6)=0.
C
C    IWORK(7)  DIMENSION OF THE INDEX 3 VARIABLES. DEFAULT IWORK(7)=0.
C
C    IWORK(8)  SWITCH FOR STEP SIZE STRATEGY
C              IF IWORK(8).EQ.1  MOD. PREDICTIVE CONTROLLER (GUSTAFSSON)
C              IF IWORK(8).EQ.2  CLASSICAL STEP SIZE CONTROL
C              THE DEFAULT VALUE (FOR IWORK(8)=0) IS IWORK(8)=1.
C              THE CHOICE IWORK(8).EQ.1 SEEMS TO PRODUCE SAFER RESULTS;
C              FOR SIMPLE PROBLEMS, THE CHOICE IWORK(8).EQ.2 PRODUCES
C              OFTEN SLIGHTLY FASTER RUNS
C
C       IF THE DIFFERENTIAL SYSTEM HAS THE SPECIAL STRUCTURE THAT
C            Y(I)' = Y(I+M2)   FOR  I=1,...,M1,
C       WITH M1 A MULTIPLE OF M2, A SUBSTANTIAL GAIN IN COMPUTERTIME
C       CAN BE ACHIEVED BY SETTING THE PARAMETERS IWORK(9) AND IWORK(10).
C       E.G., FOR SECOND ORDER SYSTEMS P'=V, V'=G(P,V), WHERE P AND V ARE 
C       VECTORS OF DIMENSION N/2, ONE HAS TO PUT M1=M2=N/2.
C       FOR M1>0 SOME OF THE INPUT PARAMETERS HAVE DIFFERENT MEANINGS:
C       - JAC: ONLY THE ELEMENTS OF THE NON-TRIVIAL PART OF THE
C              JACOBIAN HAVE TO BE STORED
C              IF (MLJAC.EQ.N-M1) THE JACOBIAN IS SUPPOSED TO BE FULL
C                 DFY(I,J) = PARTIAL F(I+M1) / PARTIAL Y(J)
C                FOR I=1,N-M1 AND J=1,N.
C              ELSE, THE JACOBIAN IS BANDED ( M1 = M2 * MM )
C                 DFY(I-J+MUJAC+1,J+K*M2) = PARTIAL F(I+M1) / PARTIAL Y(J+K*M2)
C                FOR I=1,MLJAC+MUJAC+1 AND J=1,M2 AND K=0,MM.
C       - MLJAC: MLJAC=N-M1: IF THE NON-TRIVIAL PART OF THE JACOBIAN IS FULL
C                0<=MLJAC<N-M1: IF THE (MM+1) SUBMATRICES (FOR K=0,MM)
C                     PARTIAL F(I+M1) / PARTIAL Y(J+K*M2),  I,J=1,M2
C                    ARE BANDED, MLJAC IS THE MAXIMAL LOWER BANDWIDTH
C                    OF THESE MM+1 SUBMATRICES
C       - MUJAC: MAXIMAL UPPER BANDWIDTH OF THESE MM+1 SUBMATRICES
C                NEED NOT BE DEFINED IF MLJAC=N-M1
C       - MAS: IF IMAS=0 THIS MATRIX IS ASSUMED TO BE THE IDENTITY AND
C              NEED NOT BE DEFINED. SUPPLY A DUMMY SUBROUTINE IN THIS CASE.
C              IT IS ASSUMED THAT ONLY THE ELEMENTS OF RIGHT LOWER BLOCK OF
C              DIMENSION N-M1 DIFFER FROM THAT OF THE IDENTITY MATRIX.
C              IF (MLMAS.EQ.N-M1) THIS SUBMATRIX IS SUPPOSED TO BE FULL
C                 AM(I,J) = M(I+M1,J+M1)     FOR I=1,N-M1 AND J=1,N-M1.
C              ELSE, THE MASS MATRIX IS BANDED
C                 AM(I-J+MUMAS+1,J) = M(I+M1,J+M1)
C       - MLMAS: MLMAS=N-M1: IF THE NON-TRIVIAL PART OF M IS FULL
C                0<=MLMAS<N-M1: LOWER BANDWIDTH OF THE MASS MATRIX
C       - MUMAS: UPPER BANDWIDTH OF THE MASS MATRIX
C                NEED NOT BE DEFINED IF MLMAS=N-M1
C
C    IWORK(9)  THE VALUE OF M1.  DEFAULT M1=0.
C
C    IWORK(10) THE VALUE OF M2.  DEFAULT M2=M1.
C
C ----------
C
C    WORK(1)   UROUND, THE ROUNDING UNIT, DEFAULT 1.D-16.
C
C    WORK(2)   THE SAFETY FACTOR IN STEP SIZE PREDICTION,
C              DEFAULT 0.9D0.
C
C    WORK(3)   DECIDES WHETHER THE JACOBIAN SHOULD BE RECOMPUTED;
C              INCREASE WORK(3), TO 0.1 SAY, WHEN JACOBIAN EVALUATIONS
C              ARE COSTLY. FOR SMALL SYSTEMS WORK(3) SHOULD BE SMALLER 
C              (0.001D0, SAY). NEGATIV WORK(3) FORCES THE CODE TO
C              COMPUTE THE JACOBIAN AFTER EVERY ACCEPTED STEP.     
C              DEFAULT 0.001D0.
C
C    WORK(4)   STOPPING CRITERION FOR NEWTON'S METHOD, USUALLY CHOSEN <1.
C              SMALLER VALUES OF WORK(4) MAKE THE CODE SLOWER, BUT SAFER.
C              DEFAULT MIN(0.03D0,RTOL(1)**0.5D0)
C
C    WORK(5) AND WORK(6) : IF WORK(5) < HNEW/HOLD < WORK(6), THEN THE
C              STEP SIZE IS NOT CHANGED. THIS SAVES, TOGETHER WITH A
C              LARGE WORK(3), LU-DECOMPOSITIONS AND COMPUTING TIME FOR
C              LARGE SYSTEMS. FOR SMALL SYSTEMS ONE MAY HAVE
C              WORK(5)=1.D0, WORK(6)=1.2D0, FOR LARGE FULL SYSTEMS
C              WORK(5)=0.99D0, WORK(6)=2.D0 MIGHT BE GOOD.
C              DEFAULTS WORK(5)=1.D0, WORK(6)=1.2D0 .
C
C    WORK(7)   MAXIMAL STEP SIZE, DEFAULT XEND-X.
C
C    WORK(8), WORK(9)   PARAMETERS FOR STEP SIZE SELECTION
C              THE NEW STEP SIZE IS CHOSEN SUBJECT TO THE RESTRICTION
C                 WORK(8) <= HNEW/HOLD <= WORK(9)
C              DEFAULT VALUES: WORK(8)=0.2D0, WORK(9)=8.D0
C
C-----------------------------------------------------------------------
C
C     OUTPUT PARAMETERS 
C     ----------------- 
C     X           X-VALUE FOR WHICH THE SOLUTION HAS BEEN COMPUTED
C                 (AFTER SUCCESSFUL RETURN X=XEND).
C
C     Y(N)        NUMERICAL SOLUTION AT X
C 
C     H           PREDICTED STEP SIZE OF THE LAST ACCEPTED STEP
C
C     IDID        REPORTS ON SUCCESSFULNESS UPON RETURN:
C                   IDID= 1  COMPUTATION SUCCESSFUL,
C                   IDID= 2  COMPUT. SUCCESSFUL (INTERRUPTED BY SOLOUT)
C                   IDID=-1  INPUT IS NOT CONSISTENT,
C                   IDID=-2  LARGER NMAX IS NEEDED,
C                   IDID=-3  STEP SIZE BECOMES TOO SMALL,
C                   IDID=-4  MATRIX IS REPEATEDLY SINGULAR.
C
C   IWORK(14)  NFCN    NUMBER OF FUNCTION EVALUATIONS (THOSE FOR NUMERICAL
C                      EVALUATION OF THE JACOBIAN ARE NOT COUNTED)  
C   IWORK(15)  NJAC    NUMBER OF JACOBIAN EVALUATIONS (EITHER ANALYTICALLY
C                      OR NUMERICALLY)
C   IWORK(16)  NSTEP   NUMBER OF COMPUTED STEPS
C   IWORK(17)  NACCPT  NUMBER OF ACCEPTED STEPS
C   IWORK(18)  NREJCT  NUMBER OF REJECTED STEPS (DUE TO ERROR TEST),
C                      (STEP REJECTIONS IN THE FIRST STEP ARE NOT COUNTED)
C   IWORK(19)  NDEC    NUMBER OF LU-DECOMPOSITIONS OF BOTH MATRICES
C   IWORK(20)  NSOL    NUMBER OF FORWARD-BACKWARD SUBSTITUTIONS, OF BOTH
C                      SYSTEMS; THE NSTEP FORWARD-BACKWARD SUBSTITUTIONS,
C                      NEEDED FOR STEP SIZE SELECTION, ARE NOT COUNTED
C-----------------------------------------------------------------------
C *** *** *** *** *** *** *** *** *** *** *** *** ***
C          DECLARATIONS 
C *** *** *** *** *** *** *** *** *** *** *** *** ***
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION Y(N),ATOL(*),RTOL(*),WORK(LWORK),IWORK(LIWORK)
      DIMENSION RPAR(*),IPAR(*)
      LOGICAL IMPLCT,JBAND,ARRET,STARTN,PRED, bPrint, bChangeTolerances, bAlwaysUse2ndErrorEstimate, bReport
      INTEGER nPrint, nMaxBadIte, var_index(N), nReport
      INTEGER  NFCN, NJAC, NSTEP, NACCPT, NREJCT, NFAILED, NDEC, NSOL, nLinSolveErr
      EXTERNAL FCN,JAC,MAS,SOLOUT, REPORTFUN
      
      if (nPrint > 0) then
        bPrint = .true.
      else
        bPrint = .false.
      endif
      bChangeTolerances = .false. ! do not cheat :)
      
      if (nReport > 0) then
        bReport = .true.
      else
        bReport = .false.
      endif
      
      if (nAlwaysUse2ndErrorEstimate>0) then
        bAlwaysUse2ndErrorEstimate = .true.
      else
        bAlwaysUse2ndErrorEstimate = .false.
      endif
      
      if (bPrint) then
          print*, 'nPrintInteger=', nPrintInteger
          print*, 'nMaxBadIte=', nMaxBadIte
      endif
C *** *** *** *** *** *** ***
C        SETTING THE PARAMETERS 
C *** *** *** *** *** *** ***
       NFCN=0
       NJAC=0
       NSTEP=0
       NACCPT=0
       NREJCT=0
       NFAILED=0
       NDEC=0
       NSOL=0
       nLinSolveErr=0
       ARRET=.FALSE.
C -------- UROUND   SMALLEST NUMBER SATISFYING 1.0D0+UROUND>1.0D0  
      IF (WORK(1).EQ.0.0D0) THEN
         UROUND=1.0D-16
      ELSE
         UROUND=WORK(1)
         IF (UROUND.LE.1.0D-19.OR.UROUND.GE.1.0D0) THEN
            WRITE(6,*)' COEFFICIENTS HAVE 20 DIGITS, UROUND=',WORK(1)
            ARRET=.TRUE.
         END IF
      END IF
C -------- CHECK AND CHANGE THE TOLERANCES
      EXPM=2.0D0/3.0D0
      IF (ITOL.EQ.0) THEN ! scalar tolerances
          IF (ATOL(1).LE.0.D0.OR.RTOL(1).LE.10.D0*UROUND) THEN
              WRITE (6,*) ' TOLERANCES ARE TOO SMALL'
              ARRET=.TRUE.
          ELSE
              if (bChangeTolerances) then
                  QUOT=ATOL(1)/RTOL(1)
                  RTOL(1)=0.1D0*RTOL(1)**EXPM !TODO: pourquoi ????
                  ATOL(1)=RTOL(1)*QUOT
              endif
          END IF
      ELSE ! vector tolerances
          DO I=1,N
              IF (ATOL(I).LE.0.D0.OR.RTOL(I).LE.10.D0*UROUND) THEN
                  WRITE (6,*) ' TOLERANCES(',I,') ARE TOO SMALL'
                  ARRET=.TRUE.
              ELSE
                if (bChangeTolerances) then
                    QUOT=ATOL(I)/RTOL(I)
                    RTOL(I)=0.1D0*RTOL(I)**EXPM
                    ATOL(I)=RTOL(I)*QUOT
                endif
              END IF
          END DO
      END IF
      if (bChangeTolerances .and. bPrint) print*, 'changed tolerances to rtol=', RTOL(1), ', atol=', ATOL(1)
C -------- NMAX , THE MAXIMAL NUMBER OF STEPS -----
      IF (IWORK(2).EQ.0) THEN
         NMAX=100000
      ELSE
         NMAX=IWORK(2)
         IF (NMAX.LE.0) THEN
            WRITE(6,*)' WRONG INPUT IWORK(2)=',IWORK(2)
            ARRET=.TRUE.
         END IF
      END IF
C -------- NIT    MAXIMAL NUMBER OF NEWTON ITERATIONS
      IF (IWORK(3).EQ.0) THEN
         NIT=7
      ELSE
         NIT=IWORK(3)
         IF (NIT.LE.0) THEN
            WRITE(6,*)' CURIOUS INPUT IWORK(3)=',IWORK(3)
            ARRET=.TRUE.
         END IF
      END IF
C -------- STARTN  SWITCH FOR STARTING VALUES OF NEWTON ITERATIONS
      IF(IWORK(4).EQ.0)THEN
         STARTN=.FALSE.
      ELSE
         STARTN=.TRUE.
      END IF
C -------- PARAMETER FOR DIFFERENTIAL-ALGEBRAIC COMPONENTS
      !NIND1=IWORK(5)
      !NIND2=IWORK(6)
      !NIND3=IWORK(7)
      !IF (NIND1.EQ.0) NIND1=N
      !IF (NIND1+NIND2+NIND3.NE.N) THEN
      ! WRITE(6,*)' CURIOUS INPUT FOR IWORK(5,6,7)=',NIND1,NIND2,NIND3
      ! ARRET=.TRUE.
      !END IF
C -------- PRED   STEP SIZE CONTROL
      IF(IWORK(8).LE.1)THEN
         PRED=.TRUE.
      ELSE
         PRED=.FALSE.
      END IF
C -------- PARAMETER FOR SECOND ORDER EQUATIONS
      M1=IWORK(9)
      M2=IWORK(10)
      NM1=N-M1
      IF (M1.EQ.0) M2=N
      IF (M2.EQ.0) M2=M1
      IF (M1.LT.0.OR.M2.LT.0.OR.M1+M2.GT.N) THEN
       WRITE(6,*)' CURIOUS INPUT FOR IWORK(9,10)=',M1,M2
       ARRET=.TRUE.
      END IF
C --------- SAFE     SAFETY FACTOR IN STEP SIZE PREDICTION
      IF (WORK(2).EQ.0.0D0) THEN
         SAFE=0.9D0
      ELSE
         SAFE=WORK(2)
         IF (SAFE.LE.0.001D0.OR.SAFE.GE.1.0D0) THEN
            WRITE(6,*)' CURIOUS INPUT FOR WORK(2)=',WORK(2)
            ARRET=.TRUE.
         END IF
      END IF
C ------ THET     DECIDES WHETHER THE JACOBIAN SHOULD BE RECOMPUTED;
      IF (WORK(3).EQ.0.D0) THEN
         THET=0.001D0
      ELSE
         THET=WORK(3)
         IF (THET.GE.1.0D0) THEN
            WRITE(6,*)' CURIOUS INPUT FOR WORK(3)=',WORK(3)
            ARRET=.TRUE.
         END IF
      END IF
C --- FNEWT   STOPPING CRITERION FOR NEWTON'S METHOD, USUALLY CHOSEN <1.
      TOLST=RTOL(1)
      IF (WORK(4).EQ.0.D0) THEN
         FNEWT=MAX(10*UROUND/TOLST,MIN(0.03D0,TOLST**0.5D0))
      ELSE
         FNEWT=WORK(4)
         IF (FNEWT.LE.UROUND/TOLST) THEN
            WRITE(6,*)' CURIOUS INPUT FOR WORK(4)=',WORK(4)
            ARRET=.TRUE.
         END IF
      END IF
C --- QUOT1 AND QUOT2: IF QUOT1 < HNEW/HOLD < QUOT2, STEP SIZE = CONST.
      IF (WORK(5).EQ.0.D0) THEN
         QUOT1=1.D0
      ELSE
         QUOT1=WORK(5)
      END IF
      IF (WORK(6).EQ.0.D0) THEN
         QUOT2=1.2D0
      ELSE
         QUOT2=WORK(6)
      END IF
      IF (QUOT1.GT.1.0D0.OR.QUOT2.LT.1.0D0) THEN
         WRITE(6,*)' CURIOUS INPUT FOR WORK(5,6)=',QUOT1,QUOT2
         ARRET=.TRUE.
      END IF
C -------- MAXIMAL STEP SIZE
      IF (WORK(7).EQ.0.D0) THEN
         HMAX=XEND-X
      ELSE
         HMAX=WORK(7)
      END IF 
C -------  FACL,FACR     PARAMETERS FOR STEP SIZE SELECTION
      IF(WORK(8).EQ.0.D0)THEN
         FACL=5.D0
      ELSE
         FACL=1.D0/WORK(8)
      END IF
      IF(WORK(9).EQ.0.D0)THEN
         FACR=1.D0/8.0D0
      ELSE
         FACR=1.D0/WORK(9)
      END IF
      IF (FACL.LT.1.0D0.OR.FACR.GT.1.0D0) THEN
            WRITE(6,*)' CURIOUS INPUT WORK(8,9)=',WORK(8),WORK(9)
            ARRET=.TRUE.
         END IF
C *** *** *** *** *** *** *** *** *** *** *** *** ***
C         COMPUTATION OF ARRAY ENTRIES
C *** *** *** *** *** *** *** *** *** *** *** *** ***
C ---- IMPLICIT, BANDED OR NOT ?
      IMPLCT=IMAS.NE.0
      JBAND=MLJAC.LT.NM1
C -------- COMPUTATION OF THE ROW-DIMENSIONS OF THE 2-ARRAYS ---
C -- JACOBIAN  AND  MATRICES E1, E2
      IF (JBAND) THEN
         LDJAC=MLJAC+MUJAC+1
         LDE1=MLJAC+LDJAC
      ELSE
         MLJAC=NM1
         MUJAC=NM1
         LDJAC=NM1
         LDE1=NM1
      END IF
C -- MASS MATRIX
      IF (IMPLCT) THEN
          IF (MLMAS.NE.NM1) THEN
              LDMAS=MLMAS+MUMAS+1
              IF (JBAND) THEN
                 IJOB=4
              ELSE
                 IJOB=3
              END IF
          ELSE
              MUMAS=NM1
              LDMAS=NM1
              IJOB=5
          END IF
C ------ BANDWITH OF "MAS" NOT SMALLER THAN BANDWITH OF "JAC"
          IF (MLMAS.GT.MLJAC.OR.MUMAS.GT.MUJAC) THEN
             WRITE (6,*) 'BANDWITH OF "MAS" NOT SMALLER THAN BANDWITH OF "JAC"'
            ARRET=.TRUE.
          END IF
      ELSE
          LDMAS=0
          IF (JBAND) THEN
             IJOB=2
          ELSE
             IJOB=1
             IF (N.GT.2.AND.IWORK(1).NE.0) IJOB=7
          END IF
      END IF
      LDMAS2=MAX(1,LDMAS)
C ------ HESSENBERG OPTION ONLY FOR EXPLICIT EQU. WITH FULL JACOBIAN
      IF ((IMPLCT.OR.JBAND).AND.IJOB.EQ.7) THEN
         WRITE(6,*)' HESSENBERG OPTION ONLY FOR EXPLICIT EQUATIONS WITH FULL JACOBIAN'
         ARRET=.TRUE.
      END IF
C ------- PREPARE THE ENTRY-POINTS FOR THE ARRAYS IN WORK -----
      IEZ1=21
      IEZ2=IEZ1+N
      IEZ3=IEZ2+N
      IEY0=IEZ3+N
      IESCAL=IEY0+N
      IEF1=IESCAL+N
      IEF2=IEF1+N
      IEF3=IEF2+N
      IECON=IEF3+N
      IEJAC=IECON+4*N
      IEMAS=IEJAC+N*LDJAC
      IEE1=IEMAS+NM1*LDMAS
      IEE2R=IEE1+NM1*LDE1
      IEE2I=IEE2R+NM1*LDE1
C ------ TOTAL STORAGE REQUIREMENT -----------
      ISTORE=IEE2I+NM1*LDE1-1
      IF(ISTORE.GT.LWORK)THEN
         WRITE(6,*)' INSUFFICIENT STORAGE FOR WORK, MIN. LWORK=',ISTORE
         ARRET=.TRUE.
      END IF
C ------- ENTRY POINTS FOR INTEGER WORKSPACE -----
      IEIP1=21
      IEIP2=IEIP1+NM1
      IEIPH=IEIP2+NM1
C --------- TOTAL REQUIREMENT ---------------
      ISTORE=IEIPH+NM1-1
      IF (ISTORE.GT.LIWORK) THEN
         WRITE(6,*)' INSUFF. STORAGE FOR IWORK, MIN. LIWORK=',ISTORE
         ARRET=.TRUE.
      END IF
C ------ WHEN A FAIL HAS OCCURED, WE RETURN WITH IDID=-1
      IF (ARRET) THEN
         IDID=-1
         RETURN
      END IF
C -------- CALL TO CORE INTEGRATOR ------------
      CALL RADCOR(N,FCN,X,Y,XEND,HMAX,H,RTOL,ATOL,ITOL,
     &   JAC,IJAC,MLJAC,MUJAC,MAS,MLMAS,MUMAS,SOLOUT,IOUT,IDID,
     &   NMAX,UROUND,SAFE,THET,FNEWT,QUOT1,QUOT2,NIT,IJOB,STARTN,
     &   PRED,FACL,FACR,M1,M2,NM1,
     &   IMPLCT,JBAND,LDJAC,LDE1,LDMAS2,WORK(IEZ1),WORK(IEZ2),
     &   WORK(IEZ3),WORK(IEY0),WORK(IESCAL),WORK(IEF1),WORK(IEF2),
     &   WORK(IEF3),WORK(IEJAC),WORK(IEE1),WORK(IEE2R),WORK(IEE2I),
     &   WORK(IEMAS),IWORK(IEIP1),IWORK(IEIP2),IWORK(IEIPH),
     &   WORK(IECON),NFCN,NJAC,NSTEP,NACCPT,NREJCT,NFAILED,NDEC,NSOL,nLinSolveErr,RPAR,IPAR,
     &   bPrint, nMaxBadIte, bAlwaysUse2ndErrorEstimate, var_index,
     &   REPORTFUN, bReport)
      IWORK(14)=NFCN
      IWORK(15)=NJAC
      IWORK(16)=NSTEP
      IWORK(17)=NACCPT
      IWORK(18)=NREJCT
      IWORK(19)=NDEC
      IWORK(20)=NSOL
      IWORK(21)=NFAILED
      IWORK(22)=nLinSolveErr
      
C -------- RESTORE TOLERANCES
      if (bChangeTolerances) then
          EXPM=1.0D0/EXPM
          IF (ITOL.EQ.0) THEN
                  QUOT=ATOL(1)/RTOL(1)
                  RTOL(1)=(10.0D0*RTOL(1))**EXPM
                  ATOL(1)=RTOL(1)*QUOT
          ELSE
              DO I=1,N
                  QUOT=ATOL(I)/RTOL(I)
                  RTOL(I)=(10.0D0*RTOL(I))**EXPM
                  ATOL(I)=RTOL(I)*QUOT
              END DO
          END IF
      endif
C ----------- RETURN -----------
      RETURN
      END
C
C     END OF SUBROUTINE RADAU5
C
C ***********************************************************
C
C     ! SUBROUTINE RADCOR(N,FCN,X,Y,XEND,HMAX,H,RTOL,ATOL,ITOL,
C     !&   JAC,IJAC,MLJAC,MUJAC,MAS,MLMAS,MUMAS,SOLOUT,IOUT,IDID,
C     !&   NMAX,UROUND,SAFE,THET,FNEWT,QUOT1,QUOT2,NIT,IJOB,STARTN,
C     !&   NIND1,NIND2,NIND3,PRED,FACL,FACR,M1,M2,NM1,
      SUBROUTINE RADCOR(N,FCN,X,Y,XEND,HMAX,H,RTOL,ATOL,ITOL,
     &   JAC,IJAC,MLJAC,MUJAC,MAS,MLMAS,MUMAS,SOLOUT,IOUT,IDID,
     &   NMAX,UROUND,SAFE,THET,FNEWT,QUOT1,QUOT2,NIT,IJOB,STARTN,
     &   PRED,FACL,FACR,M1,M2,NM1,
     &   IMPLCT,BANDED,LDJAC,LDE1,LDMAS,Z1,Z2,Z3,
     &   Y0,SCAL,F1,F2,F3,FJAC,E1,E2R,E2I,FMAS,IP1,IP2,IPHES,
     &   CONT,NFCN,NJAC,NSTEP,NACCPT,NREJCT,NFAILED,NDEC,NSOL,nLinSolveErr,RPAR,IPAR,
     &   bPrint, nMaxBadIte,bAlwaysUse2ndErrorEstimate, var_index,
     &   REPORTFUN, bReport)
C ----------------------------------------------------------
C     CORE INTEGRATOR FOR RADAU5
C     PARAMETERS SAME AS IN RADAU5 WITH WORKSPACE ADDED 
C ---------------------------------------------------------- 
C         DECLARATIONS 
C ---------------------------------------------------------- 
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION Y(N),Z1(N),Z2(N),Z3(N),Y0(N),SCAL(N),F1(N),F2(N),F3(N)
      DIMENSION FJAC(LDJAC,N),FMAS(LDMAS,NM1),CONT(4*N)
      DIMENSION E1(LDE1,NM1),E2R(LDE1,NM1),E2I(LDE1,NM1)
      DIMENSION ATOL(*),RTOL(*),RPAR(*),IPAR(*)
      INTEGER IP1(NM1),IP2(NM1),IPHES(NM1)
      COMMON /CONRA5/NN,NN2,NN3,NN4,XSOL,HSOL,C2M1,C1M1
      COMMON/LINAL/MLE,MUE,MBJAC,MBB,MDIAG,MDIFF,MBDIAG
      LOGICAL REJECT,FIRST,IMPLCT,BANDED,CALJAC,STARTN,CALHES, bReport
      !LOGICAL INDEX1,INDEX2,INDEX3
      LOGICAL LAST, PRED, bHighIndex
      EXTERNAL FCN, REPORTFUN
      LOGICAL bPrint, bAlwaysUse2ndErrorEstimate
      DOUBLE PRECISION RES_NORM, RES_NORM_OLD, THRES
      INTEGER nMaxBadIte, NBAD, var_index(N), var_exp(N), NFAILED, nLinSolveErr, new_nLinSolveErr
                  
C *** *** *** *** *** *** ***
C  INITIALISATIONS
C *** *** *** *** *** *** ***
C --------- DUPLIFY N FOR COMMON BLOCK CONT -----
      IF (bPrint) print*, 'Entering RADCOR'
      NN=N
      NN2=2*N
      NN3=3*N 
      LRC=4*N
C -------- CHECK THE INDEX OF THE PROBLEM ----- 
      bHighIndex = ANY( var_index > 1)
      if (bHighIndex) then
        var_exp(1:N) = max( 0, var_index - 1 )
        if (bPrint) print*, 'var_exp=', var_exp
      else
        var_exp(1:N) = 0
      endif
      !INDEX1=NIND1.NE.0
      !INDEX2=NIND2.NE.0
      !INDEX3=NIND3.NE.0
C ------- COMPUTE MASS MATRIX FOR IMPLICIT CASE ----------
      IF (IMPLCT) then
        CALL MAS(NM1,FMAS,LDMAS,RPAR,IPAR)
       ! print*, 'fresh mass matrix :)'
       ! DO I=1,min(N,MUMAS+MLMAS+1)
       !     print*, ' i=',i
       !     print*, '   FMAS(I,:)=', FMAS(I,:)
       ! END DO
       ! DO j=1,N
       !     print*, ' j=',j
       !     print*, '   FMAS(:,j)=', FMAS(:,j)
       ! END DO
        
        !stop 'debug'
        
      endif
C ---------- CONSTANTS ---------
      SQ6=DSQRT(6.D0)
      C1=(4.D0-SQ6)/10.D0
      C2=(4.D0+SQ6)/10.D0
      C1M1=C1-1.D0
      C2M1=C2-1.D0
      C1MC2=C1-C2
      DD1=-(13.D0+7.D0*SQ6)/3.D0
      DD2=(-13.D0+7.D0*SQ6)/3.D0
      DD3=-1.D0/3.D0
      U1=(6.D0+81.D0**(1.D0/3.D0)-9.D0**(1.D0/3.D0))/30.D0
      ALPH=(12.D0-81.D0**(1.D0/3.D0)+9.D0**(1.D0/3.D0))/60.D0
      BETA=(81.D0**(1.D0/3.D0)+9.D0**(1.D0/3.D0))*DSQRT(3.D0)/60.D0
      CNO=ALPH**2+BETA**2
      U1=1.0D0/U1
      ALPH=ALPH/CNO
      BETA=BETA/CNO
      T11=9.1232394870892942792D-02
      T12=-0.14125529502095420843D0
      T13=-3.0029194105147424492D-02
      T21=0.24171793270710701896D0
      T22=0.20412935229379993199D0
      T23=0.38294211275726193779D0
      T31=0.96604818261509293619D0
      TI11=4.3255798900631553510D0
      TI12=0.33919925181580986954D0
      TI13=0.54177053993587487119D0
      TI21=-4.1787185915519047273D0
      TI22=-0.32768282076106238708D0
      TI23=0.47662355450055045196D0
      TI31=-0.50287263494578687595D0
      TI32=2.5719269498556054292D0
      TI33=-0.59603920482822492497D0
      IF (M1.GT.0) IJOB=IJOB+10
      POSNEG=SIGN(1.D0,XEND-X)
      HMAXN=MIN(ABS(HMAX),ABS(XEND-X)) 
      IF (ABS(H).LE.10.D0*UROUND) H=1.0D-6
      H=MIN(ABS(H),HMAXN)
      H=SIGN(H,POSNEG)
      HOLD=H
      REJECT=.FALSE.
      FIRST=.TRUE.
      LAST=.FALSE.
      IF ((X+H*1.0001D0-XEND)*POSNEG.GE.0.D0) THEN
         H=XEND-X
         LAST=.TRUE.
      END IF
      HOPT=H
      FACCON=1.D0
      CFAC=SAFE*(1+2*NIT)
      NSING=0
      XOLD=X
      IF (IOUT.NE.0) THEN
          IRTRN=1
          NRSOL=1
          XOSOL=XOLD
          XSOL=X
          DO I=1,N
             CONT(I)=Y(I)
          END DO
          NSOLU=N
          HSOL=HOLD
          CALL SOLOUT(NRSOL,XOSOL,XSOL,Y,CONT,LRC,NSOLU,
     &                RPAR,IPAR,IRTRN)
          IF (IRTRN.LT.0) GOTO 179
      END IF
      MLE=MLJAC
      MUE=MUJAC
      MBJAC=MLJAC+MUJAC+1
      MBB=MLMAS+MUMAS+1
      MDIAG=MLE+MUE+1
      MDIFF=MLE+MUE-MUMAS
      MBDIAG=MUMAS+1
      N2=2*N
      N3=3*N
      IF (ITOL.EQ.0) THEN
        SCAL(1:N)=ATOL(1)+RTOL(1)*ABS(Y(1:N))
      ELSE
        SCAL(1:N)=ATOL(1:N)+RTOL(1:N)*ABS(Y(1:N))
      END IF
      HHFAC=H
      CALL FCN(N,X,Y,Y0,RPAR,IPAR)
      NFCN=NFCN+1
C --- BASIC INTEGRATION STEP  
  10  CONTINUE
C *** *** *** *** *** *** ***
C  COMPUTATION OF THE JACOBIAN
C *** *** *** *** *** *** ***
      NJAC=NJAC+1
      IF (bPrint) print*, 'Updating Jacobian' 
      if (bReport) CALL REPORTFUN(X,H,-10,-1,-1) ! update jacobian
      IF (IJAC.EQ.0) THEN
C --- COMPUTE JACOBIAN MATRIX NUMERICALLY
         IF (BANDED) THEN
C --- JACOBIAN IS BANDED
            MUJACP=MUJAC+1
            MD=MIN(MBJAC,M2)
            DO MM=1,M1/M2+1
               DO K=1,MD
                  J=K+(MM-1)*M2
 12               F1(J)=Y(J)
                  F2(J)=DSQRT(UROUND*MAX(1.D-5,ABS(Y(J))))
                  Y(J)=Y(J)+F2(J)
                  J=J+MD
                  IF (J.LE.MM*M2) GOTO 12 
                  CALL FCN(N,X,Y,CONT,RPAR,IPAR)
                  J=K+(MM-1)*M2
                  J1=K
                  LBEG=MAX(1,J1-MUJAC)+M1
 14               LEND=MIN(M2,J1+MLJAC)+M1
                  Y(J)=F1(J)
                  MUJACJ=MUJACP-J1-M1
                  DO L=LBEG,LEND
                     FJAC(L+MUJACJ,J)=(CONT(L)-Y0(L))/F2(J) 
                  END DO
                  J=J+MD
                  J1=J1+MD
                  LBEG=LEND+1
                  IF (J.LE.MM*M2) GOTO 14
               END DO
            END DO
         ELSE
C --- JACOBIAN IS FULL
            DO I=1,N
               YSAFE=Y(I)
               DELT=DSQRT(UROUND*MAX(1.D-5,ABS(YSAFE)))
               Y(I)=YSAFE+DELT
               CALL FCN(N,X,Y,CONT,RPAR,IPAR)
               DO J=M1+1,N
                 FJAC(J-M1,I)=(CONT(J)-Y0(J))/DELT
               END DO
               Y(I)=YSAFE
            END DO
         END IF
      ELSE
C --- COMPUTE JACOBIAN MATRIX ANALYTICALLY
         CALL JAC(N,X,Y,FJAC,LDJAC,RPAR,IPAR)
      END IF
      CALJAC=.TRUE.
      CALHES=.TRUE.
  20  CONTINUE
C --- COMPUTE THE MATRICES E1 AND E2 AND THEIR DECOMPOSITIONS
      ! they ahve the form fac1*M - Jac
      FAC1=U1/H
      ALPHN=ALPH/H
      BETAN=BETA/H
      IF (bPrint) print*, 'Decomposing matrices'
      if (bReport) CALL REPORTFUN(X,H,-11,-1,-1) ! refactor jacobian
      CALL DECOMR(N,FJAC,LDJAC,FMAS,LDMAS,MLMAS,MUMAS,
     &            M1,M2,NM1,FAC1,E1,LDE1,IP1,IER,IJOB,CALHES,IPHES)
      IF (IER.NE.0) GOTO 78
      CALL DECOMC(N,FJAC,LDJAC,FMAS,LDMAS,MLMAS,MUMAS,
     &            M1,M2,NM1,ALPHN,BETAN,E2R,E2I,LDE1,IP2,IER,IJOB)
      IF (IER.NE.0) GOTO 78
      NDEC=NDEC+1
  30  CONTINUE
      NSTEP=NSTEP+1
      if (bPrint) then
        print*, ''
        print*, 'step ', NSTEP, '(t=',X,', dt=',H,')'
      endif
      IF (NSTEP.GT.NMAX) GOTO 178
      IF (0.1D0*ABS(H).LE.ABS(X)*UROUND) GOTO 177
      
      ! Update scaling for algebraic variables
      !IF (INDEX2) THEN
      !   DO I=NIND1+1,NIND1+NIND2
      !      SCAL(I)=SCAL(I)/HHFAC
      !   END DO
      !END IF
      !IF (INDEX3) THEN
      !   DO I=NIND1+NIND2+1,NIND1+NIND2+NIND3
      !      SCAL(I)=SCAL(I)/(HHFAC*HHFAC)
      !   END DO
      !END IF
      if (bHighIndex) then
        SCAL(1:N) = SCAL(1:N) / (HHFAC ** var_exp)
      endif
      if (bPrint) print*, 'HHFAC=',HHFAC
      !if (bPrint) print*, 'SCAL=',SCAL
      XPH=X+H
C *** *** *** *** *** *** ***
C  STARTING VALUES FOR NEWTON ITERATION
C *** *** *** *** *** *** ***
      IF (FIRST.OR.STARTN) THEN
         IF (bPrint) print*, '  Trivial initial guess'
         DO I=1,N
            Z1(I)=0.D0
            Z2(I)=0.D0
            Z3(I)=0.D0
            F1(I)=0.D0
            F2(I)=0.D0
            F3(I)=0.D0
         END DO
      ELSE ! Extrapolate dense output of previous step
         IF (bPrint) print*, '  Extrapolated initial guess'
         C3Q=H/HOLD
         C1Q=C1*C3Q
         C2Q=C2*C3Q
         DO I=1,N
            AK1=CONT(I+N)
            AK2=CONT(I+N2)
            AK3=CONT(I+N3)
            Z1I=C1Q*(AK1+(C1Q-C2M1)*(AK2+(C1Q-C1M1)*AK3))
            Z2I=C2Q*(AK1+(C2Q-C2M1)*(AK2+(C2Q-C1M1)*AK3))
            Z3I=C3Q*(AK1+(C3Q-C2M1)*(AK2+(C3Q-C1M1)*AK3))
            Z1(I)=Z1I
            Z2(I)=Z2I
            Z3(I)=Z3I
            F1(I)=TI11*Z1I+TI12*Z2I+TI13*Z3I
            F2(I)=TI21*Z1I+TI22*Z2I+TI23*Z3I
            F3(I)=TI31*Z1I+TI32*Z2I+TI33*Z3I
         END DO
      END IF
C *** *** *** *** *** *** ***
C  LOOP FOR THE SIMPLIFIED NEWTON ITERATION
C *** *** *** *** *** *** ***
            NEWT=0 ! iteration counter
            NBAD=0 ! number of iterations with rate > 1 during the current Newton loop
            FACCON=MAX(FACCON,UROUND)**0.8D0
            THETA=ABS(THET)
  40        CONTINUE
            IF (bPrint) print*, '  iter', NEWT,'/',NIT

            IF (NEWT.GE.NIT) GOTO 78 ! Too many Newton iterations
            !TODO: weird, why is there no correction of dt based on the number of iterations as in the case where convergence is deemed too slow ?
            
            ! COMPUTE THE RIGHT-HAND SIDE
            CONT(1:N)=Y(1:N)+Z1(1:N)
            CALL FCN(N,X+C1*H,CONT,Z1,RPAR,IPAR)  !Z1 will be equal to FCN(t,y+z1), i.e. RHS evaluated at current first quadrature point iterate
            
            CONT(1:N)=Y(1:N)+Z2(1:N)
            CALL FCN(N,X+C2*H,CONT,Z2,RPAR,IPAR)
            
            CONT(1:N)=Y(1:N)+Z3(1:N)
            CALL FCN(N,XPH,CONT,Z3,RPAR,IPAR)
            NFCN=NFCN+3
            
            ! SOLVE THE LINEAR SYSTEMS
            DO I=1,N
              A1=Z1(I)
              A2=Z2(I)
              A3=Z3(I)
              Z1(I)=TI11*A1+TI12*A2+TI13*A3
              Z2(I)=TI21*A1+TI22*A2+TI23*A3
              Z3(I)=TI31*A1+TI32*A2+TI33*A3
            END DO
            RES_NORM_OLD = RES_NORM
            CALL SLVRAD(N,FJAC,LDJAC,MLJAC,MUJAC,FMAS,LDMAS,MLMAS,MUMAS,
     &               M1,M2,NM1,FAC1,ALPHN,BETAN,E1,E2R,E2I,LDE1,Z1,Z2,Z3,
     &               F1,F2,F3,CONT,IP1,IP2,IPHES,IER,IJOB,RES_NORM)
            NSOL=NSOL+1
            NEWT=NEWT+1
            ! Compute norm of Newton increment
            DYNO=0.D0
            DO I=1,N
               DENOM=SCAL(I)
               DYNO=DYNO+(Z1(I)/DENOM)**2+(Z2(I)/DENOM)**2
     &          +(Z3(I)/DENOM)**2
            END DO
            DYNO=DSQRT(DYNO/N3)
            IF (bPrint) print*, '    Norm of the residuals: ', RES_NORM
            IF (bPrint) print*, '    Scaled norm of the increment: ', DYNO
            
            IF (NEWT.GT.1.AND.NEWT.LT.NIT) THEN
                THQ=DYNO/DYNOLD ! current convergence rate
                THRES=RES_NORM/RES_NORM_OLD ! current convergence rate for residuals
                !TODO: properly combine rate of convergence on increment and residuals to assess convergence
                IF (NEWT.EQ.2) THEN
                   THETA=THQ
                ELSE
                   THETA=SQRT(THQ*THQOLD) ! some kind of rate smoothing ?
                END IF
                if (bPrint) then
                    print*, '    THQ=',THQ,', THETA=', THETA,', THQOLD=', THQOLD, ', THRES=', THRES
                    if (THRES>0.99) then
                        print*, 'THRES is large !'
                        if (THQ>0.99) then
                            print*, ' /!\ both residual and increment norms increase...'
                        endif
                    endif
                endif
                THQOLD=THQ
                IF (THETA.LT.0.99D0) THEN
                    FACCON=THETA/(1.0D0-THETA)
                    ! Estimated true error = FACCON * DYNO
                    DYTH=FACCON*DYNO*THETA**(NIT-1-NEWT)/FNEWT ! Estimated true error after NIT iterations (relative to Newton tolerance)
                    if (bPrint) then
                        print*, '    Estimated true error  =',FACCON*DYNO/FNEWT
                        print*, '    Estimated final error =',DYTH
                    endif

                    IF (DYTH.GE.1.0D0) THEN ! Newton will likely not converge to the desired precision within NIT iterations
                         if (NBAD<nMaxBadIte) then
                            IF (bPrint) print*, '    Bad iteration, estimated final error too large'
                            NBAD = NBAD + 1
                         else
                             if (bPrint) then
                                if (nMaxBadIte>0) print*, '    Too many bad iterations'
                                print*, '    Newton will not converge to the desired precision'
                             endif                                
                             if (bReport) CALL REPORTFUN(X,H,5,NEWT,NBAD) ! poor convergence
                             QNEWT=DMAX1(1.0D-4,DMIN1(20.0D0,DYTH))
                             HHFAC=.8D0*QNEWT**(-1.0D0/(4.0D0+NIT-1-NEWT))
                             H=HHFAC*H
                             if (bPrint) print*, '    QNEWT=',QNEWT, ', HHFAC=',HHFAC, ', H=',H
                             
                             REJECT=.TRUE.
                             LAST=.FALSE.
                             NFAILED=NFAILED+1
                             IF (CALJAC) then ! Jacobian had already been udpated at current X --> only decrease dt and refactor
                                GOTO 20 ! refactor jacobian
                             else
                                GOTO 10 ! update jacobian
                             endif
                        endif
                    END IF
                ELSE ! Newton step diverges
                    IF (THETA>10) then
                        IF (bPrint) print*, '    Newton diverges'
                        GOTO 78
                    ELSE
                        if (NBAD<nMaxBadIte) then
                            IF (bPrint) print*, '    Bad iteration, rate=',THETA,' > 0.99'
                            NBAD = NBAD + 1
                        ELSE
                            IF (bPrint) print*, '    Too many bad iterations'
                            GOTO 78
                        ENDIF
                    ENDIF
                END IF
            END IF
            DYNOLD=MAX(DYNO,UROUND)
            
            ! Apply increment
            IF (bPrint) print*, '    Applying increment'
            F1(1:N) = F1(1:N) + Z1(1:N)
            F2(1:N) = F2(1:N) + Z2(1:N)
            F3(1:N) = F3(1:N) + Z3(1:N)
            
            Z1(1:N) = T11*F1(1:N) + T12*F2(1:N) + T13*F3(1:N)
            Z2(1:N) = T21*F1(1:N) + T22*F2(1:N) + T23*F3(1:N)
            Z3(1:N) = T31*F1(1:N) +     F2(1:N)
            
            IF (FACCON*DYNO > FNEWT) THEN ! Not yet converged --> perform another Newton step
                GOTO 40
            ENDIF
      IF (bPrint) print*, '  Newton converged'
      
      !************************!
      !*** ERROR ESTIMATION ***!
      !************************!
      CALL ESTRAD (N,FJAC,LDJAC,MLJAC,MUJAC,FMAS,LDMAS,MLMAS,MUMAS,
     &          H,DD1,DD2,DD3,FCN,NFCN,Y0,Y,IJOB,X,M1,M2,NM1,
     &          E1,LDE1,Z1,Z2,Z3,CONT,F1,F2,IP1,IPHES,SCAL,ERR,
     &          FIRST,REJECT,FAC1,RPAR,IPAR,
     &          new_nLinSolveErr, bPrint, bAlwaysUse2ndErrorEstimate)

      nLinSolveErr = nLinSolveErr + new_nLinSolveErr
C --- COMPUTATION OF HNEW
C --- WE REQUIRE .2<=HNEW/H<=8.
      FAC=MIN(SAFE,CFAC/(NEWT+2*NIT))
      QUOT=MAX(FACR,MIN(FACL,ERR**.25D0/FAC))
      HNEW=H/QUOT
C *** *** *** *** *** *** ***
C  IS THE ERROR SMALL ENOUGH ?
C *** *** *** *** *** *** ***
      if (bPrint) print*, '  ERR=',ERR
      IF (ERR.LT.1.D0) THEN
C --- STEP IS ACCEPTED 
         FIRST=.FALSE.
         NACCPT=NACCPT+1
         if (bReport) CALL REPORTFUN(X,H,0,NEWT,NBAD) ! step accepted
         IF (PRED) THEN
C       --- PREDICTIVE CONTROLLER OF GUSTAFSSON
            IF (NACCPT.GT.1) THEN
               FACGUS=(HACC/H)*(ERR**2/ERRACC)**0.25D0/SAFE
               if (bPrint) print*, '  FACGUS=',FACGUS
               FACGUS=MAX(FACR,MIN(FACL,FACGUS))
               if (bPrint) print*, '    --> =',FACGUS
               QUOT=MAX(QUOT,FACGUS)
               if (bPrint) print*, '  QUOT=', QUOT
               HNEW=H/QUOT
            END IF
            HACC=H
            ERRACC=MAX(1.0D-2,ERR)
         END IF
         XOLD=X
         HOLD=H
         X=XPH 
         DO I=1,N
            Y(I)=Y(I)+Z3(I)  
            Z2I=Z2(I)
            Z1I=Z1(I)
            CONT(I+N)=(Z2I-Z3(I))/C2M1
            AK=(Z1I-Z2I)/C1MC2
            ACONT3=Z1I/C1
            ACONT3=(AK-ACONT3)/C2
            CONT(I+N2)=(AK-CONT(I+N))/C1M1
            CONT(I+N3)=CONT(I+N2)-ACONT3
         END DO
         IF (ITOL.EQ.0) THEN
            SCAL(1:N)=ATOL(1)+RTOL(1)*ABS(Y(1:N))
         ELSE
            SCAL(1:N)=ATOL(1:N)+RTOL(1:N)*ABS(Y(1:N))
         END IF
         IF (IOUT.NE.0) THEN
             NRSOL=NACCPT+1
             XSOL=X
             XOSOL=XOLD
             CONT(1:N)=Y(1:N)
             NSOLU=N
             HSOL=HOLD
             CALL SOLOUT(NRSOL,XOSOL,XSOL,Y,CONT,LRC,NSOLU,
     &                   RPAR,IPAR,IRTRN)
             IF (IRTRN.LT.0) GOTO 179
         END IF
         CALJAC=.FALSE.
         IF (LAST) THEN ! final time step has been performed
            H=HOPT
            IDID=1
            RETURN
         END IF
         CALL FCN(N,X,Y,Y0,RPAR,IPAR)
         NFCN=NFCN+1
         HNEW=POSNEG*MIN(ABS(HNEW),HMAXN)
         HOPT=HNEW
         HOPT=MIN(H,HNEW)
         IF (REJECT) HNEW=POSNEG*MIN(ABS(HNEW),ABS(H)) 
         REJECT=.FALSE.
         IF ((X+HNEW/QUOT1-XEND)*POSNEG.GE.0.D0) THEN
            H=XEND-X
            LAST=.TRUE.
         ELSE
            QT=HNEW/H 
            HHFAC=H
            IF (THETA.LE.THET.AND.QT.GE.QUOT1.AND.QT.LE.QUOT2) GOTO 30
            H=HNEW 
         END IF
         HHFAC=H
         IF (THETA.LE.THET) GOTO 20 ! refactor jacobian
         GOTO 10 !update jacobian
      ELSE
C --- STEP IS REJECTED  
         if (bReport) CALL REPORTFUN(X,H,1,NEWT,NBAD) ! step rejected
         REJECT=.TRUE.
         LAST=.FALSE.
         IF (FIRST) THEN
             H=H*0.1D0
             HHFAC=0.1D0
         ELSE 
             HHFAC=HNEW/H
             H=HNEW
         END IF
         !IF (NACCPT.GE.1) NREJCT=NREJCT+1
         NREJCT=NREJCT+1
         IF (CALJAC) GOTO 20 ! refactor jacobian
         GOTO 10 ! update jacobian
      END IF
C --- UNEXPECTED STEP-REJECTION
  78  CONTINUE
      IF (IER.NE.0) THEN
          IF (bPrint) print*, '  Singular problem'
          NSING=NSING+1
          if (bReport) CALL REPORTFUN(X,H,4,NEWT,NBAD) ! matrix is singular
          IF (NSING.GE.5) GOTO 176
      ELSE
          if (bReport) CALL REPORTFUN(X,H,6,NEWT,NBAD) ! Newton failed (bis)
      END IF
      NFAILED=NFAILED+1
      IF (bPrint) print*, '  dt will be reduced'
      H=H*0.5D0 
      HHFAC=0.5D0
      REJECT=.TRUE.
      LAST=.FALSE.
      IF (CALJAC) GOTO 20 ! refactor jacobian
      GOTO 10 ! updating jacobian
C --- FAIL EXIT
 176  CONTINUE
      WRITE(6,979)X   
      WRITE(6,*) ' MATRIX IS REPEATEDLY SINGULAR, IER=',IER
      IF (bPrint) print*, '  Matrix is repeatedly singular'
      IDID=-4
      RETURN
 177  CONTINUE
      WRITE(6,979)X   
      WRITE(6,*) ' STEP SIZE T0O SMALL, H=',H
      IF (bPrint) print*, '  step size too small, H=',H
      IDID=-3
      RETURN
 178  CONTINUE
      WRITE(6,979)X   
      WRITE(6,*) ' MORE THAN NMAX =',NMAX,'STEPS ARE NEEDED' 
      IF (bPrint) print*, '  maximum number of steps reached'
      IDID=-2
      RETURN
C --- EXIT CAUSED BY SOLOUT
 179  CONTINUE
      WRITE(6,979)X
 979  FORMAT(' EXIT OF RADAU5 AT X=',E18.4) 
      IF (bPrint) print*, '  Exit at X=',X
      IDID=2
      RETURN
      END
C
C     END OF SUBROUTINE RADCOR
C
C ***********************************************************
C
      DOUBLE PRECISION FUNCTION CONTR5(I,X,CONT,LRC) 
C ----------------------------------------------------------
C     THIS FUNCTION CAN BE USED FOR CONINUOUS OUTPUT. IT PROVIDES AN
C     APPROXIMATION TO THE I-TH COMPONENT OF THE SOLUTION AT X.
C     IT GIVES THE VALUE OF THE COLLOCATION POLYNOMIAL, DEFINED FOR
C     THE LAST SUCCESSFULLY COMPUTED STEP (BY RADAU5).
C ----------------------------------------------------------
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      DIMENSION CONT(LRC)
      COMMON /CONRA5/NN,NN2,NN3,NN4,XSOL,HSOL,C2M1,C1M1
      S=(X-XSOL)/HSOL
      CONTR5=CONT(I)+S*(CONT(I+NN)+(S-C2M1)*(CONT(I+NN2)
     &     +(S-C1M1)*CONT(I+NN3)))
      RETURN
      END
C
C     END OF FUNCTION CONTR5
C
C ***********************************************************
