// Copy pasted and adapted from :
// http://www.gnu.org/software/libc/manual/html_node/Argp-Example-3.html#Argp-Example-3

#include <argp.h>

/* This structure is used by main to communicate with parse_opt. */
struct arguments
{
  char *args[1];  /* ARG1 = hash file */
  char *format;   /* Argument for -f */
  char test;      /*Argument for -t*/
};

const char *argp_program_version =
"John ENSAE 0.1";

const char *argp_program_bug_address =
"<paul.minder@free.fr>";

/*
   OPTIONS.  Field 1 in ARGP.
   Order of fields: {NAME, KEY, ARG, FLAGS, DOC}.
*/
static struct argp_option options[] =
{
  {"format", 'f', "FORMAT", 0,
   "Parse hashes of a specific format"},
  {"test", 't', 0, 0, "Test speed"},
  {0}
};

/*
   PARSER. Field 2 in ARGP.
   Order of parameters: KEY, ARG, STATE.
*/
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  struct arguments *arguments = (struct arguments *)state->input;

  switch (key)
    {
    case 'f':
      arguments->format = arg;
      break;
    case 't':
      arguments->test = 1;
    case ARGP_KEY_ARG:
      if (state->arg_num >= 1) 
          {
            argp_usage(state);
          }
      arguments->args[state->arg_num] = arg;
      break;
    case ARGP_KEY_END:
      if (state->arg_num < 1)
          {
            argp_usage (state);
          }
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/*
   ARGS_DOC. Field 3 in ARGP.
   A description of the non-option command-line arguments
     that we accept.
*/
static char args_doc[] = "[HASHFILE]";

/*
  DOC.  Field 4 in ARGP.
  Program documentation.
*/
static char doc[] =
"John ENSAE -- A small password recovery tool by Paul "
"and Romaric from ENSAE.";

/*
   The ARGP structure itself.
*/
static struct argp argp = {options, parse_opt, args_doc, doc};
