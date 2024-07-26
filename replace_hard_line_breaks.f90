program replace_hard_line_breaks
  implicit none

  character(len=100)            :: input_file, output_file
  character(len=1000)           :: line, buffer
  integer                       :: io_status, i
  logical                       :: in_code_block

  ! Get input and output file names
  print *, "Enter input Markdown file name:"
  read *, input_file
  print *, "Enter output file name:"
  read *, output_file

  ! Open input file
  open(unit=10, file=trim(input_file), status='old', action='read', iostat=io_status)
  if (io_status /= 0) then
    print *, "Error opening input file"
    stop
  end if

  ! Open output file
  open(unit=20, file=trim(output_file), status='replace', action='write', iostat=io_status)
  if (io_status /= 0) then
    print *, "Error opening output file"
    stop
  end if

  buffer = ""
  in_code_block = .false.

  ! Process the file line by line
  do
    read(10, '(A)', iostat=io_status) line
    if (io_status /= 0) exit  ! End of file or error

    ! Check if we're entering or leaving a code block
    if (line(1:3) == '```') then
      in_code_block = .not. in_code_block
      if (len_trim(buffer) > 0) then
        write(20, '(A)') trim(buffer)
        buffer = ""
      end if
      write(20, '(A)') trim(line)
      cycle
    end if

    ! If in a code block, write the line as-is
    if (in_code_block) then
      if (len_trim(buffer) > 0) then
        write(20, '(A)') trim(buffer)
        buffer = ""
      end if
      write(20, '(A)') trim(line)
      cycle
    end if

    ! Check if the line is empty or starts with Markdown syntax
    if (len_trim(line) == 0 .or. &
        line(1:1) == '#' .or. &
        line(1:2) == '- ' .or. &
        line(1:2) == '* ' .or. &
        line(1:3) == '1. ') then
      if (len_trim(buffer) > 0) then
        write(20, '(A)') trim(buffer)
        buffer = ""
      end if
      write(20, '(A)') trim(line)
    else
      ! Append the line to the buffer, always adding a space
      if (len_trim(buffer) > 0) then
        buffer = trim(buffer) // ' ' // trim(line)
      else
        buffer = trim(line)
      end if
    end if
  end do

  ! Write any remaining content in the buffer
  if (len_trim(buffer) > 0) then
    write(20, '(A)') trim(buffer)
  end if

  ! Close files
  close(10)
  close(20)

  print *, "Processing complete. Output written to ", trim(output_file)
end program replace_hard_line_breaks

! this program replaces the hard line breaks created by llama3.1 after proofreading text
! with spaces so that the paragraphs flow properly (noticable when editing words)
! the entire program was written by Claude Sonnet 3.5 without modification
! a previous instruction failed to ask for replacement of hard line breaks with spaces
! but this version was output on the first try (with the change from the prior incorrect prompt):
!
! if (len_trim(buffer) > 0) then
!  buffer = trim(buffer) // ' ' // trim(line)
! else
!  buffer = trim(line)
! end if
!
