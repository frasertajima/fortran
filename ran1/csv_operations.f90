module csv_operations
    implicit none
    private
    public                                  :: read_csv, write_sample_csv
  
  contains
    subroutine read_csv(filename, x, y, sig, n)
      character(len=*), intent(in)          :: filename
      real, allocatable, intent(out)        :: x(:), y(:), sig(:)
      integer, intent(out)                  :: n
      integer                               :: io_stat, i
      
      ! Count number of lines in file
      open(unit=10, file=filename, status='old', action='read')
      n = 0
      do
        read(10, *, iostat=io_stat)
        if (io_stat /= 0) exit
        n = n + 1
      end do
      close(10)
      
      ! Allocate arrays
      allocate(x(n), y(n), sig(n))
      
      ! Read data from file
      open(unit=10, file=filename, status='old', action='read')
      do i = 1, n
        read(10, *, iostat=io_stat) x(i), y(i), sig(i)
        if (io_stat /= 0) then
          print *, "Error reading line ", i
          stop
        end if
      end do
      close(10)
    end subroutine read_csv
  
    subroutine write_sample_csv(filename, n, spread)
      character(len=*), intent(in)          :: filename
      integer, intent(in)                   :: n
      real, intent(in)                      :: spread
      integer                               :: i
      real                                  :: x, y, gasdev_result
      
      call init_random_seed()
      
      open(unit=10, file=filename, status='replace', action='write')
      do i = 1, n
        x = 0.1 * i
        call random_normal(gasdev_result)
        y = -2.0 * x + 1.0 + spread * gasdev_result
        write(10, '(F10.4, ",", F10.4, ",", F10.4)') x, y, spread
      end do
      close(10)
    end subroutine write_sample_csv
  
    subroutine init_random_seed()
      integer                               :: i, n, clock
      integer, dimension(:), allocatable    :: seed
      
      call random_seed(size = n)
      allocate(seed(n))
      
      call system_clock(count=clock)
      
      seed = clock + 37 * (/ (i - 1, i = 1, n) /)
      call random_seed(put = seed)
      
      deallocate(seed)
    end subroutine init_random_seed
  
    subroutine random_normal(x)
      real, intent(out)                     :: x
      real                                  :: u1, u2, r
      do
        call random_number(u1)
        call random_number(u2)
        u1 = 2.0 * u1 - 1.0
        u2 = 2.0 * u2 - 1.0
        r = u1*u1 + u2*u2
        if (r < 1.0 .and. r > 0.0) exit
      end do
      x = u1 * sqrt(-2.0 * log(r) / r)
    end subroutine random_normal
end module csv_operations
