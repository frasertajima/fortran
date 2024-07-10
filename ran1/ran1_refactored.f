      FUNCTION RAN1(IDUM)
      IMPLICIT NONE
      INTEGER IDUM
      REAL RAN1
      
      INTEGER, PARAMETER :: M1 = 259200, IA1 = 7141, IC1 = 54773
      INTEGER, PARAMETER :: M2 = 134456, IA2 = 8121, IC2 = 28411
      INTEGER, PARAMETER :: M3 = 243000, IA3 = 4561, IC3 = 51349
      REAL, PARAMETER :: RM1 = 3.8580247E-6, RM2 = 7.4373773E-6
      
      INTEGER, SAVE :: IX1, IX2, IX3, IFF = 0
      REAL, SAVE :: R(97)
      
      INTEGER :: J
      
      IF (IDUM .LT. 0 .OR. IFF .EQ. 0) THEN
        IFF = 1
        IX1 = MOD(IC1 - IDUM, M1)
        IX1 = MOD(IA1 * IX1 + IC1, M1)
        IX2 = MOD(IX1, M2)
        IX1 = MOD(IA1 * IX1 + IC1, M1)
        IX3 = MOD(IX1, M3)
        
        DO J = 1, 97
          IX1 = MOD(IA1 * IX1 + IC1, M1)
          IX2 = MOD(IA2 * IX2 + IC2, M2)
          R(J) = (REAL(IX1) + REAL(IX2) * RM2) * RM1
        END DO
        
        IDUM = 1
      END IF
      
      IX1 = MOD(IA1 * IX1 + IC1, M1)
      IX2 = MOD(IA2 * IX2 + IC2, M2)
      IX3 = MOD(IA3 * IX3 + IC3, M3)
      J = 1 + (97 * IX3) / M3
      
      IF (J .GT. 97 .OR. J .LT. 1) THEN
        WRITE(*,*) 'ERROR: J out of bounds'
        STOP
      END IF
      
      RAN1 = R(J)
      R(J) = (REAL(IX1) + REAL(IX2) * RM2) * RM1
      
      END FUNCTION RAN1
      ! produces correct results unlike original ran1
