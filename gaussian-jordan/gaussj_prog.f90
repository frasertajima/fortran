program matrix_operations
    implicit none

    integer, parameter :: np = 20
    integer :: n, m, k, l, j, io_status
    real, dimension(np,np) :: a, b, ai, x, u, t
    character(len=80) :: line

    open(unit=5, file='matrx1.dat', status='old')

    ! Skip the first line
    read(5,*)

    do
        ! Read "Size of matrix (NxN), Number of solutions:"
        read(5,'(A)', iostat=io_status) line
        if (io_status /= 0) exit  ! End of file or read error

        ! Read N and M
        read(5,*) n, m

        ! Read "Matrix A:"
        read(5,*)

        ! Read matrix A
        do k = 1, n
            read(5,*) (a(k,l), l=1,n)
        end do

        ! Read "Solution vectors:"
        read(5,*)

        ! Read matrix B
        do l = 1, m
            read(5,*) (b(k,l), k=1,n)
        end do

        ! Save matrices for later testing of results
        ai = a
        x = b

        ! Invert matrix
        call gaussj(ai, n, np, x, m, np)

        ! Add this print statement as example does not provide answer!!
        print *, 'Solution vector (x, y, z):'
        do k = 1, n
            print *, x(k, 1)
        end do

        print *, 'Inverse of Matrix A : '
        do k = 1, n
            print '(6F12.6)', (ai(k,l), l=1,n)
        end do

        ! Test results
        ! Check inverse
        print *, 'A times A-inverse (compare with unit matrix)'
        do k = 1, n
            u(k,:) = matmul(a(k,:), ai)
            print '(6F12.6)', (u(k,l), l=1,n)
        end do

        ! Check vector solutions
        print *, 'Check the following vectors for equality:'
        print '(T12,A8,T23,A12)', 'Original', 'Matrix*Sol''n'
        do l = 1, m
            print '(1X,A,I2,A)', 'Vector ', l, ':'
            t(:,l) = matmul(a, x(:,l))
            do k = 1, n
                print '(8X,2F12.6)', b(k,l), t(k,l)
            end do
        end do

        print *, '***********************************'

        ! Read "NEXT PROBLEM" or "END"
        read(5,'(A)', iostat=io_status) line
        if (io_status /= 0 .or. trim(line) == 'END') exit
    end do

    close(5)

end program matrix_operations

! correct Numeric Recipes book example
! The program now correctly reads the input file format, including skipping the
! initial description line and handling the "NEXT PROBLEM" lines between datasets.
! It uses a character variable line to read and check for "END" or "NEXT PROBLEM" lines.
! The main loop structure now uses do-exit instead of do-while, which allows for more
! flexible exit conditions. Error handling has been improved with the use of the iostat
! parameter in read statements. The program will exit the loop if it reaches the end of
! the file or encounters an "END" line.
! ADDED: section to print out the answer, namely x,y,z; the example did not
! output the actual answer, only that it was right!
