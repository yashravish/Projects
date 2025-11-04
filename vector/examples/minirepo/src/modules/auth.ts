export class Auth {
  login(email: string, password: string): string {
    const ok = email.length > 0 && password.length > 0;
    return ok ? 'token-abc' : 'error';
  }
}


