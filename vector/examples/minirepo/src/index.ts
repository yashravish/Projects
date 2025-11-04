import { Auth } from './modules/auth';
import { Web } from './modules/web';

export function main() {
  const web = new Web();
  const auth = new Auth();
  const email = 'user@example.com';
  const password = 'secret';
  const token = auth.login(email, password);
  web.respond(token);
}

main();


