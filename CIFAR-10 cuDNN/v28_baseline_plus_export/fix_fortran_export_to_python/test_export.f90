! Minimal test case for F-order to C-order conversion
! Creates a small 4D array with known values and exports it
program test_export
    implicit none
    
    ! Small test array: (2, 2, 2, 2) = (K, C, H, W)
    real(4) :: test_array(2, 2, 2, 2)
    integer :: k, c, h, w
    integer :: counter
    
    print *, "Creating test array (2,2,2,2) with sequential values..."
    
    ! Fill with sequential values for easy tracking
    counter = 1
    do k = 1, 2
        do c = 1, 2
            do h = 1, 2
                do w = 1, 2
                    test_array(k, c, h, w) = real(counter)
                    counter = counter + 1
                end do
            end do
        end do
    end do
    
    ! Print the array structure
    print *, ""
    print *, "Array structure (Fortran F-order):"
    print *, "test_array(k,c,h,w) where k,c,h,w ∈ {1,2}"
    print *, ""
    
    do k = 1, 2
        print *, "K =", k
        do c = 1, 2
            print *, "  C =", c
            do h = 1, 2
                write(*, '(A,I1,A,2F6.1)') "    H=", h, ": ", test_array(k,c,h,1), test_array(k,c,h,2)
            end do
        end do
        print *, ""
    end do
    
    ! Export as binary (F-order, as-is)
    print *, "Exporting to test_array_f_order.bin (F-order, as-is)..."
    open(unit=10, file='test_array_f_order.bin', form='unformatted', access='stream', status='replace')
    write(10) test_array
    close(10)
    
    ! Export as text for verification
    print *, "Exporting to test_array.txt (for verification)..."
    open(unit=11, file='test_array.txt', status='replace')
    write(11, '(A)') "# Test array (2,2,2,2) in F-order"
    write(11, '(A)') "# Format: k c h w value"
    do k = 1, 2
        do c = 1, 2
            do h = 1, 2
                do w = 1, 2
                    write(11, '(4I2,F8.1)') k, c, h, w, test_array(k,c,h,w)
                end do
            end do
        end do
    end do
    close(11)
    
    ! Also export with manual C-order conversion
    print *, "Exporting to test_array_c_order.bin (manual C-order conversion)..."
    block
        real(4) :: c_order(2, 2, 2, 2)
        integer :: i, j, l, m
        
        ! Convert: c_order(w,h,c,k) = test_array(k,c,h,w)
        do i = 1, 2  ! k
            do j = 1, 2  ! c
                do l = 1, 2  ! h
                    do m = 1, 2  ! w
                        c_order(m, l, j, i) = test_array(i, j, l, m)
                    end do
                end do
            end do
        end do
        
        open(unit=12, file='test_array_c_order.bin', form='unformatted', access='stream', status='replace')
        write(12) c_order
        close(12)
    end block
    
    print *, ""
    print *, "✅ Export complete!"
    print *, "Files created:"
    print *, "  - test_array_f_order.bin (F-order, as-is)"
    print *, "  - test_array_c_order.bin (manual C-order conversion)"
    print *, "  - test_array.txt (text format for verification)"
    
end program test_export
