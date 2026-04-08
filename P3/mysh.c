#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <ctype.h>

#define BUFFER_SIZE 1024
#define MAX_TOKENS 64
#define MAX_PATH_LEN 256
#define CMD_DIE "die"


const char *SEARCH_PATHS[] = {
    "/usr/local/bin/",
    "/usr/bin/",
    "/bin/"
};
#define NUM_SEARCH_PATHS (sizeof(SEARCH_PATHS) / sizeof(char *))

#define CMD_CD "cd"
#define CMD_PWD "pwd"
#define CMD_WHICH "which"
#define CMD_EXIT "exit"
#define CMD_DIE "die"

#define COND_AND "and"
#define COND_OR "or"

#define REDIR_IN "<"
#define REDIR_OUT ">"
#define PIPE_TOKEN "|"
#define COMMENT_CHAR '#'

int process_command(char *command, int *last_status);
int execute_command(char **tokens, int num_tokens, int input_fd, int output_fd, int *is_builtin);
int execute_pipeline(char **tokens, int num_tokens);
int process_builtin(char **tokens, int num_tokens, int output_fd);
char *find_program(const char *name);
void expand_wildcards(char *pattern, char ***tokens, int *num_tokens, int *capacity);
void welcome_message();
void goodbye_message();
int is_interactive();
int read_command(int fd, char *buffer, int max_size);
void tokenize_command(char *command, char ***tokens, int *num_tokens);
void free_tokens(char **tokens, int num_tokens);

int interactive_mode = 0;
int last_command_status = 0;

int main(int argc, char *argv[]) {
    int input_fd = STDIN_FILENO;
    char buffer[BUFFER_SIZE];
    int status = EXIT_SUCCESS;

    if (argc > 1) {
        input_fd = open(argv[1], O_RDONLY);
        if (input_fd < 0) {
            perror("mysh");
            return EXIT_FAILURE;
        }
    }

    interactive_mode = isatty(input_fd);

    if (interactive_mode) {
        welcome_message();
    }

    while (1) {
        if (interactive_mode) {
            printf("mysh> ");
            fflush(stdout);
        }

        int bytes_read = read_command(input_fd, buffer, BUFFER_SIZE);
        
        if (bytes_read <= 0) {
            break;
        }

        int cmd_status = process_command(buffer, &last_command_status);
        if (cmd_status < 0) {
            status = (cmd_status == -1) ? EXIT_SUCCESS : EXIT_FAILURE;
            break;
        }
    }

    if (interactive_mode) {
        goodbye_message();
    }

    if (input_fd != STDIN_FILENO) {
        close(input_fd);
    }

    return status;
}

void welcome_message() {
    printf("Welcome to my shell!\n");
}

void goodbye_message() {
    printf("mysh: exiting\n");
}

int is_interactive() {
    return interactive_mode;
}

int read_command(int fd, char *buffer, int max_size) {
    int bytes_read = 0;
    int total_read = 0;
    char c;

    memset(buffer, 0, max_size);

    while (total_read < max_size - 1) {
        bytes_read = read(fd, &c, 1);
        
        if (bytes_read <= 0) {
            if (total_read == 0) {
                return bytes_read;
            }
            break;
        }

        buffer[total_read++] = c;

        if (c == '\n') {
            break;
        }
    }

    buffer[total_read] = '\0';
    
    return total_read;
}

char *my_strdup(const char *s) {
    size_t len = strlen(s) + 1;
    char *new_str = malloc(len);
    if (new_str == NULL) return NULL;
    return memcpy(new_str, s, len);
}

void tokenize_command(char *command, char ***tokens, int *num_tokens) {
    char *token;
    int capacity = MAX_TOKENS;
    
    *tokens = (char **)malloc(capacity * sizeof(char *));
    if (*tokens == NULL) {
        perror("malloc");
        exit(EXIT_FAILURE);
    }
    
    *num_tokens = 0;
    
    token = strtok(command, " \t\n");
    
    if (token != NULL && token[0] == COMMENT_CHAR) {
        return; 
    }
    
    while (token != NULL) {
        if (token[0] == COMMENT_CHAR) {
            break;
        }
        
        if (strcmp(token, REDIR_IN) == 0 || 
            strcmp(token, REDIR_OUT) == 0 || 
            strcmp(token, PIPE_TOKEN) == 0) {
            
            (*tokens)[(*num_tokens)++] = my_strdup(token);
            
        } else {
            int i = 0;
            int start = 0;
            
            while (token[i] != '\0') {
                if (token[i] == '<' || token[i] == '>' || token[i] == '|') {
                    if (i > start) {
                        char temp = token[i];
                        token[i] = '\0';
                        (*tokens)[(*num_tokens)++] = my_strdup(token + start);
                        token[i] = temp;
                    }
                    
                    char special[2] = {token[i], '\0'};
                    (*tokens)[(*num_tokens)++] = my_strdup(special);
                    
                    start = i + 1;
                }
                i++;
            }
            
            if (token[start] != '\0') {
                (*tokens)[(*num_tokens)++] = my_strdup(token + start);
            }
        }
        
        if (*num_tokens >= capacity - 1) {
            capacity *= 2;
            *tokens = (char **)realloc(*tokens, capacity * sizeof(char *));
            if (*tokens == NULL) {
                perror("realloc");
                exit(EXIT_FAILURE);
            }
        }
        
        token = strtok(NULL, " \t\n");
    }
    
    (*tokens)[*num_tokens] = NULL;
}

void free_tokens(char **tokens, int num_tokens) {
    for (int i = 0; i < num_tokens; i++) {
        free(tokens[i]);
    }
    free(tokens);
}

int match_pattern(const char *pattern, const char *filename) {
    const char *p = pattern;
    const char *f = filename;
    
    if (p[0] == '*' && f[0] == '.') {
        return 0;
    }
    
    const char *star = strchr(p, '*');
    if (star == NULL) {
        return strcmp(p, f) == 0; 
    }
    
    size_t prefix_len = star - p;
    const char *suffix = star + 1;
    size_t suffix_len = strlen(suffix);
    size_t filename_len = strlen(f);
    
    if (filename_len < prefix_len + suffix_len) {
        return 0;
    }
    
    if (strncmp(p, f, prefix_len) != 0) {
        return 0;
    }
    
    if (suffix_len > 0 && strcmp(suffix, f + filename_len - suffix_len) != 0) {
        return 0;
    }
    
    return 1;
}

void expand_wildcards(char *pattern, char ***tokens, int *num_tokens, int *capacity) {
    if (strchr(pattern, '*') == NULL) {
        (*tokens)[(*num_tokens)++] = my_strdup(pattern);
        return;
    }
    
    char dir_path[MAX_PATH_LEN] = ".";
    char file_pattern[MAX_PATH_LEN];
    
    char *slash = strrchr(pattern, '/');
    if (slash != NULL) {
        int dir_len = slash - pattern;
        strncpy(dir_path, pattern, dir_len);
        dir_path[dir_len] = '\0';
        strcpy(file_pattern, slash + 1);
    } else {
        strcpy(file_pattern, pattern);
    }
    
    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        (*tokens)[(*num_tokens)++] = my_strdup(pattern);
        return;
    }
    
    int found_match = 0;
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (match_pattern(file_pattern, entry->d_name)) {
            found_match = 1;
            
            char full_path[MAX_PATH_LEN];
            if (slash != NULL) {
                size_t dir_len = strlen(dir_path);
                size_t name_len = strlen(entry->d_name);
                
                if (dir_len + name_len + 2 > MAX_PATH_LEN) {
                    continue;
                }
                
                strcpy(full_path, dir_path);
                strcat(full_path, "/");
                strcat(full_path, entry->d_name);
            } else {
                strcpy(full_path, entry->d_name);
            }
            
            (*tokens)[(*num_tokens)++] = my_strdup(full_path);
            
            if (*num_tokens >= *capacity - 1) {
                *capacity *= 2;
                *tokens = (char **)realloc(*tokens, *capacity * sizeof(char *));
                if (*tokens == NULL) {
                    perror("realloc");
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    
    if (!found_match) {
        (*tokens)[(*num_tokens)++] = my_strdup(pattern);
    }
    
    closedir(dir);
}

char *find_program(const char *name) {
    if (strchr(name, '/') != NULL) {
        return my_strdup(name);
    }
    
    if (strcmp(name, CMD_CD) == 0 || 
        strcmp(name, CMD_PWD) == 0 || 
        strcmp(name, CMD_WHICH) == 0 || 
        strcmp(name, CMD_EXIT) == 0 || 
        strcmp(name, CMD_DIE) == 0) {
        return my_strdup(name);
    }
    
    for (int i = 0; i < NUM_SEARCH_PATHS; i++) {
        char full_path[MAX_PATH_LEN];
        snprintf(full_path, MAX_PATH_LEN, "%s%s", SEARCH_PATHS[i], name);
        
        if (access(full_path, X_OK) == 0) {
            return my_strdup(full_path);
        }
    }
    
    return NULL;
}

int process_builtin(char **tokens, int num_tokens, int output_fd) {
    if (strcmp(tokens[0], CMD_CD) == 0) {
        if (num_tokens != 2) {
            write(STDERR_FILENO, "cd: wrong number of arguments\n", 30);
            return 1; 
        }
        
        if (chdir(tokens[1]) != 0) {
            write(STDERR_FILENO, "cd: No such file or directory\n", 30);
            return 1; 
        }
        
        return 0; 
    } 
    else if (strcmp(tokens[0], CMD_PWD) == 0) {
        char cwd[MAX_PATH_LEN];
        if (getcwd(cwd, MAX_PATH_LEN) == NULL) {
            perror("pwd");
            return 1; 
        }
        
        char buffer[MAX_PATH_LEN + 2];
        snprintf(buffer, sizeof(buffer), "%s\n", cwd);
        write(output_fd, buffer, strlen(buffer));
        return 0; 
    } 
    else if (strcmp(tokens[0], CMD_WHICH) == 0) {
        if (num_tokens != 2) {
            return 1; 
        }
        
        if (strcmp(tokens[1], CMD_CD) == 0 || 
            strcmp(tokens[1], CMD_PWD) == 0 || 
            strcmp(tokens[1], CMD_WHICH) == 0 || 
            strcmp(tokens[1], CMD_EXIT) == 0 || 
            strcmp(tokens[1], CMD_DIE) == 0) {
            return 1; 
        }
        
        char *program_path = find_program(tokens[1]);
        if (program_path == NULL) {
            return 1; 
        }
        
        char buffer[MAX_PATH_LEN + 2];
        snprintf(buffer, sizeof(buffer), "%s\n", program_path);
        write(output_fd, buffer, strlen(buffer));
        free(program_path);
        return 0;
    } 
    else if (strcmp(tokens[0], CMD_EXIT) == 0) {
        return -1; 
    } 
    else if (strcmp(tokens[0], CMD_DIE) == 0) {
        for (int i = 1; i < num_tokens; i++) {
            write(STDOUT_FILENO, tokens[i], strlen(tokens[i]));
            if (i < num_tokens - 1) {
                write(STDOUT_FILENO, " ", 1);
            }
        }
        if (num_tokens > 1) {
            write(STDOUT_FILENO, "\n", 1);
        }
        
        return -2; 
    }
    
    return 2; 
}

int execute_command(char **tokens, int num_tokens, int input_fd, int output_fd, int *is_builtin) {
    if (num_tokens == 0) {
        *is_builtin = 0;
        return 0; 
    }
    
    int builtin_result = process_builtin(tokens, num_tokens, output_fd);
    if (builtin_result != 2) {
        *is_builtin = 1;
        return builtin_result;
    }
    
    *is_builtin = 0;
    
    
    char *program_path = find_program(tokens[0]);
    if (program_path == NULL) {
        fprintf(stderr, "mysh: command not found: %s\n", tokens[0]);
        return 1; 
    }
    
    pid_t pid = fork();
    
    if (pid < 0) {
        perror("fork");
        free(program_path);
        return 1; 
    } 
    else if (pid == 0) {

        if (input_fd != STDIN_FILENO) {
            if (dup2(input_fd, STDIN_FILENO) < 0) {
                perror("dup2");
                exit(EXIT_FAILURE);
            }
            close(input_fd);
        }
        
        
        if (output_fd != STDOUT_FILENO) {
            if (dup2(output_fd, STDOUT_FILENO) < 0) {
                perror("dup2");
                exit(EXIT_FAILURE);
            }
            close(output_fd);
        }
        
        if (!interactive_mode && input_fd == STDIN_FILENO) {
            close(STDIN_FILENO);
        }
        
        execv(program_path, tokens);
        
        perror("execv");
        exit(EXIT_FAILURE);
    } 
    else {
       
        if (input_fd != STDIN_FILENO) {
            close(input_fd);
        }
        if (output_fd != STDOUT_FILENO) {
            close(output_fd);
        }
        
        int status;
        waitpid(pid, &status, 0);
        
        free(program_path);
        
        if (WIFEXITED(status)) {
            return WEXITSTATUS(status);
        } else {
            return 1; 
        }
    }
}

int execute_pipeline(char **tokens, int num_tokens) {
    int pipe_index = -1;
    for (int i = 0; i < num_tokens; i++) {
        if (strcmp(tokens[i], PIPE_TOKEN) == 0) {
            pipe_index = i;
            break;
        }
    }
    
    if (pipe_index < 0) {
        return 1; 
    }
    
    tokens[pipe_index] = NULL;
    
    int pipefd[2];
    if (pipe(pipefd) < 0) {
        perror("pipe");
        return 1; 
    }
    
    int is_builtin1;
    int status1 = execute_command(tokens, pipe_index, STDIN_FILENO, pipefd[1], &is_builtin1);
    
    if (is_builtin1 && (status1 == -1 || status1 == -2)) {
        close(pipefd[0]);
        close(pipefd[1]);
        return status1;
    }
    
    int is_builtin2;
    int status2 = execute_command(&tokens[pipe_index + 1], num_tokens - pipe_index - 1, pipefd[0], STDOUT_FILENO, &is_builtin2);
    
    if (is_builtin2 && (status2 == -1 || status2 == -2)) {
        return status2;
    }
    
    return status2;
}

int process_command(char *command, int *last_status) {
    char **tokens = NULL;
    int num_tokens = 0;
    int result = 0;
    
    tokenize_command(command, &tokens, &num_tokens);
    
    if (num_tokens == 0) {
        free_tokens(tokens, num_tokens);
        return 0;
    }
    
    if (strcmp(tokens[0], COND_AND) == 0) {
        if (*last_status != 0) {
            free_tokens(tokens, num_tokens);
            return 0;
        }
        free(tokens[0]); 
        for (int i = 0; i < num_tokens - 1; i++) {
            tokens[i] = tokens[i+1];
        }
        tokens[num_tokens - 1] = NULL;
        num_tokens--;
    }
    else if (strcmp(tokens[0], COND_OR) == 0) {
        if (*last_status == 0) {
            free_tokens(tokens, num_tokens);
            return 0;
        }
        free(tokens[0]); 
        for (int i = 0; i < num_tokens - 1; i++) {
            tokens[i] = tokens[i+1];
        }
        tokens[num_tokens - 1] = NULL;
        num_tokens--;
    }
    
    char **expanded_tokens = (char **)malloc(MAX_TOKENS * sizeof(char *));
    int expanded_num = 0;
    int expanded_capacity = MAX_TOKENS;
    
    for (int i = 0; i < num_tokens; i++) {
        if ((i > 0 && (strcmp(tokens[i-1], REDIR_IN) == 0 || strcmp(tokens[i-1], REDIR_OUT) == 0))) {
            expanded_tokens[expanded_num++] = my_strdup(tokens[i]);
        } else {
            expand_wildcards(tokens[i], &expanded_tokens, &expanded_num, &expanded_capacity);
        }
    }
    
    free(tokens);
    
    tokens = expanded_tokens;
    num_tokens = expanded_num;
    expanded_tokens[num_tokens] = NULL;
    
    int has_pipe = 0;
    for (int i = 0; i < num_tokens; i++) {
        if (strcmp(tokens[i], PIPE_TOKEN) == 0) {
            has_pipe = 1;
            break;
        }
    }
    
    if (has_pipe) {
        result = execute_pipeline(tokens, num_tokens);
    } else {
        int input_fd = STDIN_FILENO;
        int output_fd = STDOUT_FILENO;
        char **cmd_tokens = (char **)malloc((num_tokens + 1) * sizeof(char *));
        int cmd_num = 0;
        
        for (int i = 0; i < num_tokens; i++) {
            if (strcmp(tokens[i], REDIR_IN) == 0) {
                if (i + 1 < num_tokens) {
                    input_fd = open(tokens[i+1], O_RDONLY);
                    if (input_fd < 0) {
                        perror("open");
                        free_tokens(tokens, num_tokens);
                        free(cmd_tokens);
                        *last_status = 1;
                        return 0;
                    }
                    i++; 
                }
            } 
            else if (strcmp(tokens[i], REDIR_OUT) == 0) {
                if (i + 1 < num_tokens) {
                    output_fd = open(tokens[i+1], O_WRONLY | O_CREAT | O_TRUNC, 0640);
                    if (output_fd < 0) {
                        perror("open");
                        if (input_fd != STDIN_FILENO)
                            close(input_fd);
                        free_tokens(tokens, num_tokens);
                        free(cmd_tokens);
                        *last_status = 1;
                        return 0;
                    }
                    i++; 
                }
            } 
            else {
                cmd_tokens[cmd_num++] = tokens[i];
            }
        }
        cmd_tokens[cmd_num] = NULL;
        
        int is_builtin;
        result = execute_command(cmd_tokens, cmd_num, input_fd, output_fd, &is_builtin);
        free(cmd_tokens);
    }
    
    if (result >= 0) {
        *last_status = result;
        result = 0;
    }
    
    free_tokens(tokens, num_tokens);
    return result;
}