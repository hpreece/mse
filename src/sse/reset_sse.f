      SUBROUTINE reset_sse_state()
*
* Reset all SSE/BSE Fortran common block state that persists
* between runs. Called from reset_interface() in C++.
*
* Fixes state leakage identified in Phase 1.1 investigation
* (2026-03-21): the RAND3 shuffle table, SINGLE output arrays,
* and TSTEPC timestep control were not being reset.
*
      IMPLICIT NONE
*
* RAND3: ran3() random number generator internal state
      INTEGER idum2,iy,ir(32)
      COMMON /RAND3/ idum2,iy,ir
*
* SINGLE: SSE output arrays
      REAL*8 scm(50000,14),spp(20,3)
      COMMON /SINGLE/ scm,spp
*
* TSTEPC: timestep control
      REAL*8 dmmax,drmax
      COMMON /TSTEPC/ dmmax,drmax
*
      INTEGER i,j
*
* Reset RAND3 to initial state (matches DATA statement in ran3.f)
      idum2 = 123456789
      iy = 0
      DO i = 1,32
         ir(i) = 0
      ENDDO
*
* Reset TSTEPC
      dmmax = 0.d0
      drmax = 0.d0
*
* Reset SINGLE (zero out)
      DO i = 1,50000
         DO j = 1,14
            scm(i,j) = 0.d0
         ENDDO
      ENDDO
      DO i = 1,20
         DO j = 1,3
            spp(i,j) = 0.d0
         ENDDO
      ENDDO
*
      RETURN
      END
