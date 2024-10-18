module radix_sort_module
  use, intrinsic :: iso_c_binding
  implicit none

contains
  subroutine radixsort(arr, n) bind(c, name='radixsort')
    integer(c_int64_t), intent(inout)   :: arr(n)
    integer(c_int), value, intent(in)   :: n
    integer(c_int64_t)                  :: max_element, exp, i, digit
    integer                             :: max_digits
    integer(c_int64_t)                  :: count(0:9), output(n)

    ! Find the maximum number to determine the number of digits
    max_element = maxval(abs(arr))
    max_digits = floor(log10(real(max_element, kind=8))) + 1

    ! Perform counting sort for every digit
    exp = 1_c_int64_t
    do while (max_digits > 0)
      ! Initialize the count array
      count = 0

      ! Store the count of occurrences in count
      do i = 1, n
        digit = mod(arr(i) / exp, 10_c_int64_t)
        count(digit) = count(digit) + 1
      end do

      ! Calculate cumulative count
      do i = 1, 9
        count(i) = count(i) + count(i-1)
      end do

      ! Build the output array
      do i = n, 1, -1
        digit = mod(arr(i) / exp, 10_c_int64_t)
        output(count(digit)) = arr(i)
        count(digit) = count(digit) - 1
      end do

      ! Copy the output array to arr
      arr = output

      exp = exp * 10_c_int64_t
      max_digits = max_digits - 1
    end do
  end subroutine radixsort
end module radix_sort_module
! gfortran -shared -fPIC -o libradixsort.so radix_sort.f90
! to create module that is called in python notebooks:
! see radix_sort_python.ipynb
