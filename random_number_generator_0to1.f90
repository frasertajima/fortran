program random_number_0to1

    use iso_fortran_env, only: int32
    implicit none
    
    integer(int32), parameter                               :: total_random_numbers = 100
    integer(int32)                                          :: num, ios, i
    character(len=100)                                      :: output
    character(len=200)                                      :: command

    print '(A)', "YubiKey generated random number from 0.00000000 to 1.000000 using 512 bits."
    print '(A)', "----------------------------------------------------------------------------------------------------------"
    print '(A)', " "
    num = 0

    do while (num < total_random_numbers)
        command = 'echo ""scd random 512"" | ' // &
          'gpg-connect-agent | ' // &
          'tr -dc 0-9 | ' // &
          'awk ''{printf "0.%10d\n", $1}'''                                             ! note the two single quotes '', not "
        call execute_command_line(command, cmdstat=ios, cmdmsg=output)
        
        if (ios /= 0) then
            print '(A,A)', "Error executing command: ", trim(output)
            stop
        end if
        
        num = num + 1
        print '(A,$)', trim(output)
    end do

end program random_number_0to1
! modified from random_password
! basis for alternative to curand_m module using YubiKey instead of iso_c_binding library for montecarlo.cuf?